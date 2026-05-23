#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#define main sample_yolov8_npu_image_main
#include "sampleYOLOV8_npu.cpp"
#undef main

#include "VideoCapture.h"

using namespace std;

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

int main(int argc, char *argv[])
{
    const char *modelPath = "../model/yolov8s_static.om";
    string rtspUrl;
    int32_t modelWidth = 640;
    int32_t modelHeight = 640;
    int32_t frameLimit = 0;
    size_t maxLogDetPerFrame = 3;
    int32_t deviceId = 1;
    uint32_t transport = RTSP_TRANS_TCP;
    int32_t statsIntervalSec = 5;

    if (argc >= 2)
    {
        rtspUrl = argv[1];
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

    if (rtspUrl.empty())
    {
        ACLLITE_LOG_ERROR("usage: %s <rtsp_url|video_file> [model_path] [model_width] [model_height] [frame_limit] [max_log_det] [device_id] [tcp|udp] [stats_interval_sec]",
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

    ACLLITE_LOG_INFO("use model: %s, input size: %d x %d, stream: %s, frameLimit=%d, maxLogDet=%zu, device=%d, transport=%s, statsIntervalSec=%d",
                     modelPath, modelWidth, modelHeight, rtspUrl.c_str(), frameLimit,
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

    ::VideoCapture videoCapture(rtspUrl, deviceId);
    AclLiteError aclRet = videoCapture.Open();
    if (aclRet != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("open stream failed, errorCode is %d", aclRet);
        return FAILED;
    }
    aclRet = videoCapture.Set(RTSP_TRANSPORT, transport);
    if (aclRet != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("set rtsp transport failed, errorCode is %d", aclRet);
        return FAILED;
    }

    double pullDecodeSumMs = 0.0;
    double preprocessSumMs = 0.0;
    double inferenceSumMs = 0.0;
    double d2hSumMs = 0.0;
    double postprocessSumMs = 0.0;
    size_t processedCount = 0;
    double intervalPullDecodeSumMs = 0.0;
    double intervalPreprocessSumMs = 0.0;
    double intervalInferenceSumMs = 0.0;
    double intervalD2hSumMs = 0.0;
    double intervalPostprocessSumMs = 0.0;
    size_t intervalCount = 0;
    std::chrono::time_point<std::chrono::steady_clock> intervalStart = std::chrono::steady_clock::now();

    while (frameLimit <= 0 || static_cast<int32_t>(processedCount) < frameLimit)
    {
        ImageData decodedFrame;
        std::chrono::time_point<std::chrono::steady_clock> pullStart = std::chrono::steady_clock::now();
        aclRet = videoCapture.Read(decodedFrame);
        std::chrono::time_point<std::chrono::steady_clock> pullEnd = std::chrono::steady_clock::now();
        double pullDecodeMs = std::chrono::duration<double, std::milli>(pullEnd - pullStart).count();
        if (aclRet != ACLLITE_OK)
        {
            ACLLITE_LOG_ERROR("read decoded frame failed or stream ended, errorCode is %d", aclRet);
            break;
        }

        std::vector<InferenceOutput> inferOutputs;
        double preprocessMs = 0.0;
        double inferenceMs = 0.0;
        double d2hMs = 0.0;
        double postprocessMs = 0.0;
        double visualizeMs = 0.0;
        double saveImageMs = 0.0;

        std::chrono::time_point<std::chrono::steady_clock> preprocessStart = std::chrono::steady_clock::now();
        ret = sampleYOLO.ProcessDecodedFrame(decodedFrame);
        std::chrono::time_point<std::chrono::steady_clock> preprocessEnd = std::chrono::steady_clock::now();
        preprocessMs = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessDecodedFrame failed, errorCode is %d", ret);
            videoCapture.Close();
            return FAILED;
        }

        std::chrono::time_point<std::chrono::steady_clock> inferStart = std::chrono::steady_clock::now();
        ret = sampleYOLO.Inference(inferOutputs);
        std::chrono::time_point<std::chrono::steady_clock> inferEnd = std::chrono::steady_clock::now();
        inferenceMs = std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            videoCapture.Close();
            return FAILED;
        }

        ret = sampleYOLO.GetResult(inferOutputs, "", processedCount, false,
                                   d2hMs, postprocessMs, visualizeMs, saveImageMs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            videoCapture.Close();
            return FAILED;
        }

        pullDecodeSumMs += pullDecodeMs;
        preprocessSumMs += preprocessMs;
        inferenceSumMs += inferenceMs;
        d2hSumMs += d2hMs;
        postprocessSumMs += postprocessMs;
        ++processedCount;
        intervalPullDecodeSumMs += pullDecodeMs;
        intervalPreprocessSumMs += preprocessMs;
        intervalInferenceSumMs += inferenceMs;
        intervalD2hSumMs += d2hMs;
        intervalPostprocessSumMs += postprocessMs;
        ++intervalCount;

        if (statsIntervalSec > 0)
        {
            std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
            double elapsedSec = std::chrono::duration<double>(now - intervalStart).count();
            if (elapsedSec >= static_cast<double>(statsIntervalSec) && intervalCount > 0)
            {
                double avgTotalMs = (intervalPullDecodeSumMs + intervalPreprocessSumMs +
                                     intervalInferenceSumMs + intervalD2hSumMs +
                                     intervalPostprocessSumMs) /
                                    intervalCount;
                double avgFps = elapsedSec > 0.0 ? static_cast<double>(intervalCount) / elapsedSec : 0.0;
                ACLLITE_LOG_INFO("periodic average in last %.2fs (%zu frames): pull_decode=%.3fms, preprocess=%.3fms, inference=%.3fms, d2h=%.3fms, postprocess=%.3fms, total=%.3fms, fps=%.3f",
                                 elapsedSec, intervalCount,
                                 intervalPullDecodeSumMs / intervalCount,
                                 intervalPreprocessSumMs / intervalCount,
                                 intervalInferenceSumMs / intervalCount,
                                 intervalD2hSumMs / intervalCount,
                                 intervalPostprocessSumMs / intervalCount,
                                 avgTotalMs, avgFps);

                intervalPullDecodeSumMs = 0.0;
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

    if (processedCount > 0)
    {
        double avgTotalMs = (pullDecodeSumMs + preprocessSumMs + inferenceSumMs +
                             d2hSumMs + postprocessSumMs) /
                            processedCount;
        double avgFps = avgTotalMs > 0.0 ? 1000.0 / avgTotalMs : 0.0;
        ACLLITE_LOG_INFO("average timing(ms) for %zu frames: pull_decode=%.3f, preprocess=%.3f, inference=%.3f, d2h=%.3f, postprocess=%.3f, total=%.3f, fps=%.3f",
                         processedCount,
                         pullDecodeSumMs / processedCount,
                         preprocessSumMs / processedCount,
                         inferenceSumMs / processedCount,
                         d2hSumMs / processedCount,
                         postprocessSumMs / processedCount,
                         avgTotalMs,
                         avgFps);
    }
    else
    {
        ACLLITE_LOG_ERROR("no frame processed from stream: %s", rtspUrl.c_str());
        return FAILED;
    }

    return SUCCESS;
}
