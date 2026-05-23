#include <acl/acl.h>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define main sample_yolov8_npu_image_main
#include "sampleYOLOV8_npu.cpp"
#undef main

#include "VideoCapture.h"

using namespace std;

struct StreamConfig
{
    string url;
    const char *modelPath;
    int32_t modelWidth;
    int32_t modelHeight;
    int32_t frameLimit;
    size_t maxLogDetPerFrame;
    int32_t deviceId;
    uint32_t transport;
    int32_t statsIntervalSec;
};

static uint32_t ParseRtspTransport(const char *value)
{
    if (value == nullptr)
    {
        return RTSP_TRANS_TCP;
    }

    string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text == "udp" || text == "0")
    {
        return RTSP_TRANS_UDP;
    }
    return RTSP_TRANS_TCP;
}

static vector<string> SplitStreams(const string &streamList)
{
    vector<string> streams;
    string item;
    stringstream ss(streamList);
    while (getline(ss, item, ','))
    {
        size_t first = item.find_first_not_of(" \t\r\n");
        size_t last = item.find_last_not_of(" \t\r\n");
        if (first != string::npos && last != string::npos)
        {
            streams.push_back(item.substr(first, last - first + 1));
        }
    }
    return streams;
}

static int RunOneStreamThread(const StreamConfig &cfg, size_t streamIndex,
                              SampleYOLOV8 &sampleYOLO, aclrtContext aclContext,
                              mutex &modelMutex)
{
    aclError setCtxRet = aclrtSetCurrentContext(aclContext);
    if (setCtxRet != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("[stream %zu] set acl context failed, errorCode is %d", streamIndex, setCtxRet);
        return FAILED;
    }

    ::VideoCapture videoCapture(cfg.url, cfg.deviceId, aclContext);
    AclLiteError aclRet = videoCapture.Open();
    if (aclRet != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("[stream %zu] open stream failed, errorCode is %d, url=%s",
                          streamIndex, aclRet, cfg.url.c_str());
        return FAILED;
    }
    aclRet = videoCapture.Set(RTSP_TRANSPORT, cfg.transport);
    if (aclRet != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("[stream %zu] set rtsp transport failed, errorCode is %d", streamIndex, aclRet);
        videoCapture.Close();
        return FAILED;
    }

    double pullDecodeSumMs = 0.0;
    double waitModelSumMs = 0.0;
    double preprocessSumMs = 0.0;
    double inferenceSumMs = 0.0;
    double d2hSumMs = 0.0;
    double postprocessSumMs = 0.0;
    size_t processedCount = 0;

    double intervalPullDecodeSumMs = 0.0;
    double intervalWaitModelSumMs = 0.0;
    double intervalPreprocessSumMs = 0.0;
    double intervalInferenceSumMs = 0.0;
    double intervalD2hSumMs = 0.0;
    double intervalPostprocessSumMs = 0.0;
    size_t intervalCount = 0;
    std::chrono::time_point<std::chrono::steady_clock> intervalStart = std::chrono::steady_clock::now();

    while (cfg.frameLimit <= 0 || static_cast<int32_t>(processedCount) < cfg.frameLimit)
    {
        ImageData decodedFrame;
        std::chrono::time_point<std::chrono::steady_clock> pullStart = std::chrono::steady_clock::now();
        aclRet = videoCapture.Read(decodedFrame);
        std::chrono::time_point<std::chrono::steady_clock> pullEnd = std::chrono::steady_clock::now();
        double pullDecodeMs = std::chrono::duration<double, std::milli>(pullEnd - pullStart).count();
        if (aclRet != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("[stream %zu] read decoded frame failed or stream ended, errorCode is %d",
                              streamIndex, aclRet);
            break;
        }

        std::vector<InferenceOutput> inferOutputs;
        double preprocessMs = 0.0;
        double inferenceMs = 0.0;
        double d2hMs = 0.0;
        double postprocessMs = 0.0;
        double visualizeMs = 0.0;
        double saveImageMs = 0.0;

        std::chrono::time_point<std::chrono::steady_clock> waitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(modelMutex);
        std::chrono::time_point<std::chrono::steady_clock> waitEnd = std::chrono::steady_clock::now();
        double waitModelMs = std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();

        setCtxRet = aclrtSetCurrentContext(aclContext);
        if (setCtxRet != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("[stream %zu] reset acl context failed, errorCode is %d", streamIndex, setCtxRet);
            videoCapture.Close();
            return FAILED;
        }

        std::chrono::time_point<std::chrono::steady_clock> preprocessStart = std::chrono::steady_clock::now();
        Result ret = sampleYOLO.ProcessDecodedFrame(decodedFrame);
        std::chrono::time_point<std::chrono::steady_clock> preprocessEnd = std::chrono::steady_clock::now();
        preprocessMs = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("[stream %zu] ProcessDecodedFrame failed, errorCode is %d", streamIndex, ret);
            videoCapture.Close();
            return FAILED;
        }

        std::chrono::time_point<std::chrono::steady_clock> inferStart = std::chrono::steady_clock::now();
        ret = sampleYOLO.Inference(inferOutputs);
        std::chrono::time_point<std::chrono::steady_clock> inferEnd = std::chrono::steady_clock::now();
        inferenceMs = std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("[stream %zu] Inference failed, errorCode is %d", streamIndex, ret);
            videoCapture.Close();
            return FAILED;
        }

        ret = sampleYOLO.GetResult(inferOutputs, "", processedCount, false,
                                   d2hMs, postprocessMs, visualizeMs, saveImageMs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("[stream %zu] GetResult failed, errorCode is %d", streamIndex, ret);
            videoCapture.Close();
            return FAILED;
        }
        lock.unlock();

        pullDecodeSumMs += pullDecodeMs;
        waitModelSumMs += waitModelMs;
        preprocessSumMs += preprocessMs;
        inferenceSumMs += inferenceMs;
        d2hSumMs += d2hMs;
        postprocessSumMs += postprocessMs;
        ++processedCount;

        intervalPullDecodeSumMs += pullDecodeMs;
        intervalWaitModelSumMs += waitModelMs;
        intervalPreprocessSumMs += preprocessMs;
        intervalInferenceSumMs += inferenceMs;
        intervalD2hSumMs += d2hMs;
        intervalPostprocessSumMs += postprocessMs;
        ++intervalCount;

        if (cfg.statsIntervalSec > 0)
        {
            std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
            double elapsedSec = std::chrono::duration<double>(now - intervalStart).count();
            if (elapsedSec >= static_cast<double>(cfg.statsIntervalSec) && intervalCount > 0)
            {
                double avgTotalMs = (intervalPullDecodeSumMs + intervalWaitModelSumMs +
                                     intervalPreprocessSumMs + intervalInferenceSumMs +
                                     intervalD2hSumMs + intervalPostprocessSumMs) /
                                    intervalCount;
                double avgFps = elapsedSec > 0.0 ? static_cast<double>(intervalCount) / elapsedSec : 0.0;
                ACLLITE_LOG_INFO("[stream %zu] periodic average in last %.2fs (%zu frames): pull_decode=%.3fms, wait_model=%.3fms, preprocess=%.3fms, inference=%.3fms, d2h=%.3fms, postprocess=%.3fms, total=%.3fms, fps=%.3f",
                                 streamIndex, elapsedSec, intervalCount,
                                 intervalPullDecodeSumMs / intervalCount,
                                 intervalWaitModelSumMs / intervalCount,
                                 intervalPreprocessSumMs / intervalCount,
                                 intervalInferenceSumMs / intervalCount,
                                 intervalD2hSumMs / intervalCount,
                                 intervalPostprocessSumMs / intervalCount,
                                 avgTotalMs, avgFps);

                intervalPullDecodeSumMs = 0.0;
                intervalWaitModelSumMs = 0.0;
                intervalPreprocessSumMs = 0.0;
                intervalInferenceSumMs = 0.0;
                intervalD2hSumMs = 0.0;
                intervalPostprocessSumMs = 0.0;
                intervalCount = 0;
                intervalStart = now;
            }
        }
    }

    videoCapture.Close();

    if (processedCount == 0)
    {
        ACLLITE_LOG_ERROR("[stream %zu] no frame processed from stream: %s", streamIndex, cfg.url.c_str());
        return FAILED;
    }

    double avgTotalMs = (pullDecodeSumMs + waitModelSumMs + preprocessSumMs + inferenceSumMs +
                         d2hSumMs + postprocessSumMs) /
                        processedCount;
    double avgFps = avgTotalMs > 0.0 ? 1000.0 / avgTotalMs : 0.0;
    ACLLITE_LOG_INFO("[stream %zu] average timing(ms) for %zu frames: pull_decode=%.3f, wait_model=%.3f, preprocess=%.3f, inference=%.3f, d2h=%.3f, postprocess=%.3f, total=%.3f, fps=%.3f, url=%s",
                     streamIndex, processedCount,
                     pullDecodeSumMs / processedCount,
                     waitModelSumMs / processedCount,
                     preprocessSumMs / processedCount,
                     inferenceSumMs / processedCount,
                     d2hSumMs / processedCount,
                     postprocessSumMs / processedCount,
                     avgTotalMs, avgFps, cfg.url.c_str());
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *modelPath = "../model/yolov8s_static.om";
    string streamList;
    int32_t modelWidth = 640;
    int32_t modelHeight = 640;
    int32_t frameLimit = 0;
    size_t maxLogDetPerFrame = 3;
    int32_t deviceId = 1;
    uint32_t transport = RTSP_TRANS_TCP;
    int32_t statsIntervalSec = 5;

    if (argc >= 2)
    {
        streamList = argv[1];
    }
    if (argc >= 3)
    {
        modelPath = argv[2];
    }
    if (argc >= 4)
    {
        modelWidth = std::atoi(argv[3]);
        modelHeight = modelWidth;
    }
    if (argc >= 5)
    {
        modelHeight = std::atoi(argv[4]);
    }
    if (argc >= 6)
    {
        frameLimit = std::atoi(argv[5]);
    }
    if (argc >= 7)
    {
        int logCount = std::atoi(argv[6]);
        maxLogDetPerFrame = (logCount > 0) ? static_cast<size_t>(logCount) : 0;
    }
    if (argc >= 8)
    {
        deviceId = std::atoi(argv[7]);
    }
    if (argc >= 9)
    {
        transport = ParseRtspTransport(argv[8]);
    }
    if (argc >= 10)
    {
        statsIntervalSec = std::atoi(argv[9]);
    }

    vector<string> streams = SplitStreams(streamList);
    if (streams.empty())
    {
        ACLLITE_LOG_ERROR("usage: %s <rtsp_url1[,rtsp_url2,...]|video_file1[,video_file2,...]> [model_path] [model_width] [model_height] [frame_limit] [max_log_det] [device_id] [tcp|udp] [stats_interval_sec]",
                          argv[0]);
        return FAILED;
    }
    if (modelWidth <= 0 || modelHeight <= 0 ||
        (modelWidth % 32) != 0 || (modelHeight % 32) != 0)
    {
        ACLLITE_LOG_ERROR("invalid input size: %d x %d, width/height must be multiples of 32",
                          modelWidth, modelHeight);
        return FAILED;
    }

    ACLLITE_LOG_INFO("thread mode: streams=%zu, model=%s, input size=%d x %d, frameLimit=%d, maxLogDet=%zu, device=%d, transport=%s, statsIntervalSec=%d",
                     streams.size(), modelPath, modelWidth, modelHeight, frameLimit,
                     maxLogDetPerFrame, deviceId,
                     transport == RTSP_TRANS_UDP ? "udp" : "tcp",
                     statsIntervalSec);

    SampleYOLOV8 sampleYOLO(modelPath, modelWidth, modelHeight, false, maxLogDetPerFrame, deviceId);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    aclrtContext aclContext = sampleYOLO.GetAclContext();
    if (aclContext == nullptr)
    {
        ACLLITE_LOG_ERROR("shared acl context is null");
        return FAILED;
    }

    mutex modelMutex;
    vector<thread> workers;
    vector<int> results(streams.size(), FAILED);
    for (size_t i = 0; i < streams.size(); ++i)
    {
        StreamConfig cfg;
        cfg.url = streams[i];
        cfg.modelPath = modelPath;
        cfg.modelWidth = modelWidth;
        cfg.modelHeight = modelHeight;
        cfg.frameLimit = frameLimit;
        cfg.maxLogDetPerFrame = maxLogDetPerFrame;
        cfg.deviceId = deviceId;
        cfg.transport = transport;
        cfg.statsIntervalSec = statsIntervalSec;

        workers.push_back(thread([cfg, i, &sampleYOLO, aclContext, &modelMutex, &results]() {
            results[i] = RunOneStreamThread(cfg, i, sampleYOLO, aclContext, modelMutex);
        }));
    }

    for (size_t i = 0; i < workers.size(); ++i)
    {
        workers[i].join();
    }

    int failedCount = 0;
    for (size_t i = 0; i < results.size(); ++i)
    {
        if (results[i] != SUCCESS)
        {
            ++failedCount;
        }
    }

    if (failedCount > 0)
    {
        ACLLITE_LOG_ERROR("thread mode finished with %d failed streams out of %zu", failedCount, streams.size());
        return FAILED;
    }

    ACLLITE_LOG_INFO("thread mode finished successfully, streams=%zu", streams.size());
    return SUCCESS;
}
