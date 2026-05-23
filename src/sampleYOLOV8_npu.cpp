#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "label.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace std;
using namespace cv;
typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct BoundBox
{
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

float ClampProbability(float value)
{
    const float epsilon = 1e-6f;
    if (value < epsilon)
    {
        return epsilon;
    }
    if (value > (1.0f - epsilon))
    {
        return 1.0f - epsilon;
    }
    return value;
}

float Sigmoid(float value)
{
    if (value >= 16.0f)
    {
        return 1.0f;
    }
    if (value <= -16.0f)
    {
        return 0.0f;
    }
    return 1.0f / (1.0f + std::exp(-value));
}

float Logit(float probability)
{
    float clippedProbability = ClampProbability(probability);
    return std::log(clippedProbability / (1.0f - clippedProbability));
}

bool ParseBoolArg(const char *value, bool defaultValue)
{
    if (value == nullptr)
    {
        return defaultValue;
    }

    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text == "1" || text == "true" || text == "yes" || text == "on")
    {
        return true;
    }
    if (text == "0" || text == "false" || text == "no" || text == "off")
    {
        return false;
    }
    return defaultValue;
}

bool ParseYoloOutputShape(const aclmdlIODims &dims, size_t &channelCount, size_t &boxCount)
{
    std::vector<size_t> validDims;
    for (size_t i = 0; i < dims.dimCount; ++i)
    {
        if (dims.dims[i] > 1)
        {
            validDims.push_back(static_cast<size_t>(dims.dims[i]));
        }
    }

    if (validDims.size() < 2)
    {
        return false;
    }

    std::pair<std::vector<size_t>::const_iterator, std::vector<size_t>::const_iterator> minmaxDims =
        std::minmax_element(validDims.begin(), validDims.end());
    channelCount = *minmaxDims.first;
    boxCount = *minmaxDims.second;
    return (channelCount > 4) && (boxCount > 0);
}

bool ParseNpuDetOutputShape(const aclmdlIODims &dims, size_t &detCount, size_t &attrCount, bool &detFirstLayout)
{
    detCount = 0;
    attrCount = 0;
    detFirstLayout = true;
    auto isAttrDim = [](size_t dim) -> bool
    {
        return (dim >= 6) && (dim <= 8);
    };

    std::vector<size_t> validDims;
    for (size_t i = 0; i < dims.dimCount; ++i)
    {
        if (dims.dims[i] > 1)
        {
            validDims.push_back(static_cast<size_t>(dims.dims[i]));
        }
    }
    if (validDims.size() < 2)
    {
        return false;
    }

    bool found = false;
    for (size_t i = 0; i + 1 < validDims.size(); ++i)
    {
        size_t dim0 = validDims[i];
        size_t dim1 = validDims[i + 1];
        if (isAttrDim(dim1) && dim0 > 0)
        {
            detCount = dim0;
            attrCount = dim1;
            detFirstLayout = true;
            found = true;
        }
        else if (isAttrDim(dim0) && dim1 > 0)
        {
            detCount = dim1;
            attrCount = dim0;
            detFirstLayout = false;
            found = true;
        }
    }
    return found;
}

class SampleYOLOV8
{
public:
    SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight,
                 bool enableDrawResult, size_t maxLogDetPerImage, int32_t deviceId);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result ProcessDecodedFrame(ImageData &decodedFrame);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release,
                     double &d2hMs, double &postprocessMs, double &visualizeMs, double &saveImageMs);
    aclrtContext GetAclContext();
    ~SampleYOLOV8();

private:
    size_t CalculateBoxNum(int32_t modelWidth, int32_t modelHeight) const;
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    int32_t srcWidth_;
    int32_t srcHeight_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
    size_t modelOutputBoxNum_;
    size_t modelOutputChannelNum_;
    // 模型输出数据类型, 用于区分 fp16/fp32 读取
    aclDataType outputDataType_;
    // 是否是图内 Decode+NMS 的 NPU 后处理输出
    bool npuPostprocessOutputEnabled_;
    size_t npuDetNum_;
    size_t npuAttrNum_;
    bool npuDetFirstLayout_;
    bool enableDrawResult_;
    size_t maxLogDetPerImage_;
};

SampleYOLOV8::SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight,
                           bool enableDrawResult, size_t maxLogDetPerImage, int32_t deviceId)
    : aclResource_(deviceId, ""), modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight), modelOutputBoxNum_(0),
      srcWidth_(modelWidth), srcHeight_(modelHeight),
      modelOutputChannelNum_(0), outputDataType_(ACL_FLOAT), npuPostprocessOutputEnabled_(false),
      npuDetNum_(0), npuAttrNum_(0), npuDetFirstLayout_(true),
      enableDrawResult_(enableDrawResult), maxLogDetPerImage_(maxLogDetPerImage)
{
}

SampleYOLOV8::~SampleYOLOV8()
{
    ReleaseResource();
}

aclrtContext SampleYOLOV8::GetAclContext()
{
    return aclResource_.GetContext();
}

size_t SampleYOLOV8::CalculateBoxNum(int32_t modelWidth, int32_t modelHeight) const
{
    size_t s8 = (modelWidth / 8) * (modelHeight / 8);
    size_t s16 = (modelWidth / 16) * (modelHeight / 16);
    size_t s32 = (modelWidth / 32) * (modelHeight / 32);
    return s8 + s16 + s32;
}

Result SampleYOLOV8::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }

    if (modelWidth_ <= 0 || modelHeight_ <= 0 ||
        (modelWidth_ % 32) != 0 || (modelHeight_ % 32) != 0)
    {
        ACLLITE_LOG_ERROR("invalid model input size: %d x %d, width/height must be multiples of 32",
                          modelWidth_, modelHeight_);
        return FAILED;
    }

    modelOutputBoxNum_ = CalculateBoxNum(modelWidth_, modelHeight_);
    modelOutputChannelNum_ = 4 + sizeof(label) / sizeof(label[0]);
    ACLLITE_LOG_INFO("model input size: %d x %d, output boxes auto calculated: %zu",
                     modelWidth_, modelHeight_, modelOutputBoxNum_);

    // 读取模型输出元信息, 自动识别输出 dtype 和输出 shape
    std::vector<ModelOutputInfo> outputInfos;
    ret = model_.GetModelOutputInfo(outputInfos);
    if (ret == ACLLITE_OK && !outputInfos.empty())
    {
        const ModelOutputInfo &out0 = outputInfos[0];
        outputDataType_ = out0.dataType;
        ACLLITE_LOG_INFO("output[0] dataType=%d, format=%d, dimCount=%zu",
                         static_cast<int>(out0.dataType),
                         static_cast<int>(out0.format),
                         static_cast<size_t>(out0.dims.dimCount));

        size_t parsedChannelNum = 0;
        size_t parsedBoxNum = 0;
        if (ParseYoloOutputShape(out0.dims, parsedChannelNum, parsedBoxNum))
        {
            modelOutputChannelNum_ = parsedChannelNum;
            modelOutputBoxNum_ = parsedBoxNum;
        }

        size_t parsedDetNum = 0;
        size_t parsedAttrNum = 0;
        bool detFirstLayout = true;
        if (ParseNpuDetOutputShape(out0.dims, parsedDetNum, parsedAttrNum, detFirstLayout))
        {
            npuPostprocessOutputEnabled_ = true;
            npuDetNum_ = parsedDetNum;
            npuAttrNum_ = parsedAttrNum;
            npuDetFirstLayout_ = detFirstLayout;
            ACLLITE_LOG_INFO("NPU postprocess output enabled: layout=%s, detNum=%zu, attrNum=%zu",
                             npuDetFirstLayout_ ? "[N,A]" : "[A,N]", npuDetNum_, npuAttrNum_);
        }
        else
        {
            ACLLITE_LOG_ERROR("NPU postprocess output not detected. This program requires NPU full pipeline (Decode+NMS in graph). Current output is not compatible.");
            ACLLITE_LOG_ERROR("Please convert model to output Nx6/Nx7(or 6xN/7xN) after NPU Decode+NMS.");
            ACLLITE_LOG_INFO("fallback to CPU decode+NMS");
        }
    }
    else
    {
        ACLLITE_LOG_WARNING("GetModelOutputInfo failed, use default output decode config");
    }

    ACLLITE_LOG_INFO("decode config: channelNum=%zu, boxNum=%zu, dataType=%d",
                     modelOutputChannelNum_, modelOutputBoxNum_, static_cast<int>(outputDataType_));

    return SUCCESS;
}

Result SampleYOLOV8::ProcessInput(string testImgPath)
{
    // read image from file
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }
    srcWidth_ = static_cast<int32_t>(image.width);
    srcHeight_ = static_cast<int32_t>(image.height);

    // copy image from host to dvpp
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8::ProcessDecodedFrame(ImageData &decodedFrame)
{
    if (decodedFrame.data == nullptr || decodedFrame.width == 0 || decodedFrame.height == 0)
    {
        ACLLITE_LOG_ERROR("decoded frame is invalid, width=%u, height=%u, data=%p",
                          decodedFrame.width, decodedFrame.height, decodedFrame.data.get());
        return FAILED;
    }

    srcWidth_ = static_cast<int32_t>(decodedFrame.width);
    srcHeight_ = static_cast<int32_t>(decodedFrame.height);

    AclLiteError ret = imageProcess_.Resize(resizedImage_, decodedFrame, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize decoded frame failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference only; output device-to-host copy is counted in postprocess.
    ret = model_.ExecuteV2(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8::GetResult(std::vector<InferenceOutput> &inferOutputs,
                               string imagePath, size_t imageIndex, bool release,
                               double &d2hMs, double &postprocessMs, double &visualizeMs, double &saveImageMs)
{
    d2hMs = 0.0;
    postprocessMs = 0.0;
    visualizeMs = 0.0;
    saveImageMs = 0.0;
    std::chrono::time_point<std::chrono::steady_clock> d2hStart = std::chrono::steady_clock::now();
    uint32_t outputDataBufId = 0;
    if (inferOutputs.size() <= outputDataBufId)
    {
        ACLLITE_LOG_ERROR("model output is empty");
        return FAILED;
    }
    for (size_t i = 0; i < inferOutputs.size(); ++i)
    {
        void *hostData = CopyDataToHost(inferOutputs[i].data.get(), inferOutputs[i].size, runMode_, MEMORY_NORMAL);
        if (hostData == nullptr)
        {
            ACLLITE_LOG_ERROR("copy model output %zu to host failed", i);
            return FAILED;
        }
        inferOutputs[i].data = SHARED_PTR_U8_BUF(hostData);
    }
    std::chrono::time_point<std::chrono::steady_clock> d2hEnd = std::chrono::steady_clock::now();
    d2hMs = std::chrono::duration<double, std::milli>(d2hEnd - d2hStart).count();

#ifdef YOLOV8_RTSP_PROCESS_SKIP_POSTPROCESS
    // RTSP process profiling mode: keep D2H, skip CPU decode/NMS/draw/save.
    return SUCCESS;
#endif

    std::chrono::time_point<std::chrono::steady_clock> postStart = std::chrono::steady_clock::now();

    // confidence threshold
    float confidenceThreshold = 0.35;
    // 输出 buffer 的真实元素字节数由 dtype 决定
    size_t outputElementSize = sizeof(float);
    if (outputDataType_ == ACL_FLOAT16)
    {
        outputElementSize = sizeof(aclFloat16);
    }
    else if (outputDataType_ != ACL_FLOAT)
    {
        ACLLITE_LOG_ERROR("unsupported output data type: %d", static_cast<int>(outputDataType_));
        return FAILED;
    }
    size_t outputElementNum = inferOutputs[outputDataBufId].size / outputElementSize;

    std::vector<float> convertedOutput;
    const float *outputData = nullptr;
    if (outputDataType_ == ACL_FLOAT)
    {
        outputData = static_cast<const float *>(inferOutputs[outputDataBufId].data.get());
    }
    else if (outputDataType_ == ACL_FLOAT16)
    {
        convertedOutput.resize(outputElementNum);
        const aclFloat16 *halfOutput = static_cast<const aclFloat16 *>(inferOutputs[outputDataBufId].data.get());
        for (size_t i = 0; i < outputElementNum; ++i)
        {
            convertedOutput[i] = aclFloat16ToFloat(halfOutput[i]);
        }
        outputData = convertedOutput.data();
    }

    if (outputData == nullptr)
    {
        ACLLITE_LOG_ERROR("output data is null after conversion");
        return FAILED;
    }

    int srcWidth = srcWidth_ > 0 ? srcWidth_ : modelWidth_;
    int srcHeight = srcHeight_ > 0 ? srcHeight_ : modelHeight_;
    auto clampFloat = [](float value, float low, float high) -> float
    {
        if (value < low)
        {
            return low;
        }
        if (value > high)
        {
            return high;
        }
        return value;
    };
    const size_t builtinLabelNum = sizeof(label) / sizeof(label[0]);
    size_t classNum = builtinLabelNum;
    bool useBuiltinLabel = true;
    vector<BoundBox> result;

    if (npuPostprocessOutputEnabled_)
    {
        size_t expectedElements = npuDetNum_ * npuAttrNum_;
        if (npuDetNum_ == 0 || npuAttrNum_ < 6 || outputElementNum < expectedElements)
        {
            ACLLITE_LOG_ERROR("invalid NPU postprocess output layout, detNum=%zu, attrNum=%zu, elementNum=%zu",
                              npuDetNum_, npuAttrNum_, outputElementNum);
            return FAILED;
        }

        auto getNpuValue = [&](size_t detIdx, size_t attrIdx) -> float
        {
            size_t flatIdx = npuDetFirstLayout_
                                 ? (detIdx * npuAttrNum_ + attrIdx)
                                 : (attrIdx * npuDetNum_ + detIdx);
            return outputData[flatIdx];
        };

        bool normalizedCoordinate = true;
        size_t sampleCount = std::min(npuDetNum_, static_cast<size_t>(64));
        for (size_t i = 0; i < sampleCount; ++i)
        {
            float x1 = std::fabs(getNpuValue(i, 0));
            float y1 = std::fabs(getNpuValue(i, 1));
            float x2 = std::fabs(getNpuValue(i, 2));
            float y2 = std::fabs(getNpuValue(i, 3));
            float maxValue = std::max(std::max(x1, y1), std::max(x2, y2));
            if (maxValue > 2.0f)
            {
                normalizedCoordinate = false;
                break;
            }
        }

        for (size_t i = 0; i < npuDetNum_; ++i)
        {
            float x1 = getNpuValue(i, 0);
            float y1 = getNpuValue(i, 1);
            float x2 = getNpuValue(i, 2);
            float y2 = getNpuValue(i, 3);
            float score = getNpuValue(i, 4);
            float classValue = getNpuValue(i, 5);
            if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2) ||
                !std::isfinite(score) || !std::isfinite(classValue))
            {
                continue;
            }
            score = (score < 0.0f || score > 1.0f) ? Sigmoid(score) : score;
            if (score < confidenceThreshold)
            {
                continue;
            }
            if (normalizedCoordinate)
            {
                x1 *= srcWidth;
                x2 *= srcWidth;
                y1 *= srcHeight;
                y2 *= srcHeight;
            }
            else
            {
                x1 = x1 * srcWidth / modelWidth_;
                x2 = x2 * srcWidth / modelWidth_;
                y1 = y1 * srcHeight / modelHeight_;
                y2 = y2 * srcHeight / modelHeight_;
            }

            float left = clampFloat(std::min(x1, x2), 0.0f, static_cast<float>(std::max(srcWidth - 1, 0)));
            float right = clampFloat(std::max(x1, x2), 0.0f, static_cast<float>(std::max(srcWidth - 1, 0)));
            float top = clampFloat(std::min(y1, y2), 0.0f, static_cast<float>(std::max(srcHeight - 1, 0)));
            float bottom = clampFloat(std::max(y1, y2), 0.0f, static_cast<float>(std::max(srcHeight - 1, 0)));
            if (right <= left || bottom <= top)
            {
                continue;
            }

            int classIndex = static_cast<int>(std::round(classValue));
            if (classIndex < 0)
            {
                continue;
            }

            BoundBox box;
            box.x = (left + right) * 0.5f;
            box.y = (top + bottom) * 0.5f;
            box.width = right - left;
            box.height = bottom - top;
            box.score = score;
            box.classIndex = static_cast<size_t>(classIndex);
            box.index = i;
            result.push_back(box);
        }
        for (size_t i = 0; i < result.size(); ++i)
        {
            if (result[i].classIndex >= builtinLabelNum)
            {
                useBuiltinLabel = false;
                break;
            }
        }
        ACLLITE_LOG_INFO("NPU postprocess boxes parsed, result size is %ld", result.size());
    }
    else
    {
        size_t offset = 4;
        size_t outputChannelNum = modelOutputChannelNum_;
        size_t outputBoxNum = modelOutputBoxNum_;
        if ((outputChannelNum == 0) && (outputBoxNum != 0) && ((outputElementNum % outputBoxNum) == 0))
        {
            outputChannelNum = outputElementNum / outputBoxNum;
        }
        if ((outputBoxNum == 0) && (outputChannelNum != 0) && ((outputElementNum % outputChannelNum) == 0))
        {
            outputBoxNum = outputElementNum / outputChannelNum;
        }
        if ((outputChannelNum <= offset) || (outputBoxNum == 0))
        {
            ACLLITE_LOG_ERROR("invalid output layout, output size %u bytes, channels %zu, boxes %zu",
                              inferOutputs[outputDataBufId].size, outputChannelNum, outputBoxNum);
            return FAILED;
        }

        classNum = outputChannelNum - offset;
        if (outputElementNum < (outputChannelNum * outputBoxNum))
        {
            ACLLITE_LOG_ERROR("output buffer is too small, elementNum=%zu, required=%zu",
                              outputElementNum, outputChannelNum * outputBoxNum);
            return FAILED;
        }

        useBuiltinLabel = (classNum == builtinLabelNum);

        auto evaluateLayout = [&](bool channelFirst) -> std::pair<double, double>
        {
            size_t sampleBoxNum = std::min(outputBoxNum, static_cast<size_t>(256));
            size_t sampleClassNum = std::min(classNum, static_cast<size_t>(16));
            if ((sampleBoxNum == 0) || (sampleClassNum == 0))
            {
                return std::make_pair(0.0, std::numeric_limits<double>::infinity());
            }
            size_t inRangeCount = 0;
            double absMean = 0.0;
            size_t sampleTotal = sampleBoxNum * sampleClassNum;
            for (size_t i = 0; i < sampleBoxNum; ++i)
            {
                for (size_t j = 0; j < sampleClassNum; ++j)
                {
                    float value = channelFirst ? outputData[(offset + j) * outputBoxNum + i]
                                               : outputData[i * outputChannelNum + (offset + j)];
                    if ((value >= 0.0f) && (value <= 1.0f))
                    {
                        ++inRangeCount;
                    }
                    absMean += std::fabs(static_cast<double>(value));
                }
            }
            return std::make_pair(static_cast<double>(inRangeCount) / static_cast<double>(sampleTotal),
                                  absMean / static_cast<double>(sampleTotal));
        };

        // 优先按参考工程使用 [C, N] 解码；如果统计特征明显更像 [N, C]，则自动切换。
        bool decodeChannelFirst = true;
        std::pair<double, double> channelFirstStat = evaluateLayout(true);
        std::pair<double, double> channelLastStat = evaluateLayout(false);
        if (channelLastStat.first > (channelFirstStat.first + 0.2))
        {
            decodeChannelFirst = false;
        }
        else if (std::fabs(channelLastStat.first - channelFirstStat.first) <= 0.2 &&
                 channelLastStat.second < (channelFirstStat.second * 0.5))
        {
            decodeChannelFirst = false;
        }
        auto getOutputValue = [&](size_t boxIdx, size_t elemIdx) -> float
        {
            size_t flatIdx = decodeChannelFirst
                                 ? (elemIdx * outputBoxNum + boxIdx)
                                 : (boxIdx * outputChannelNum + elemIdx);
            return outputData[flatIdx];
        };

        bool applySigmoidToScores = false;
        for (size_t i = 0; i < outputBoxNum && !applySigmoidToScores; ++i)
        {
            for (size_t j = 0; j < classNum; ++j)
            {
                float value = getOutputValue(i, offset + j);
                if ((value < 0.0f) || (value > 1.0f))
                {
                    applySigmoidToScores = true;
                    break;
                }
            }
        }
        float scoreThresholdRaw = applySigmoidToScores ? Logit(confidenceThreshold) : confidenceThreshold;

        // filter boxes by confidence threshold
        vector<BoundBox> boxes;
        size_t yIndex = 1;
        size_t widthIndex = 2;
        size_t heightIndex = 3;

        for (size_t i = 0; i < outputBoxNum; ++i)
        {
            float maxRawValue = -std::numeric_limits<float>::infinity();
            size_t maxIndex = 0;
            for (size_t j = 0; j < classNum; ++j)
            {
                float value = getOutputValue(i, offset + j);
                if (value > maxRawValue)
                {
                    // index of class
                    maxIndex = j;
                    maxRawValue = value;
                }
            }

            if (maxRawValue > scoreThresholdRaw)
            {
                BoundBox box;
                box.x = getOutputValue(i, 0) * srcWidth / modelWidth_;
                box.y = getOutputValue(i, yIndex) * srcHeight / modelHeight_;
                box.width = getOutputValue(i, widthIndex) * srcWidth / modelWidth_;
                box.height = getOutputValue(i, heightIndex) * srcHeight / modelHeight_;
                box.score = applySigmoidToScores ? Sigmoid(maxRawValue) : maxRawValue;
                box.classIndex = maxIndex;
                box.index = i;
                if (maxIndex < classNum)
                {
                    boxes.push_back(box);
                }
            }
        }

        // filter boxes by NMS
        result.clear();
        float NMSThreshold = 0.45;
        int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
        std::sort(boxes.begin(), boxes.end(), sortScore);
        BoundBox boxMax;
        BoundBox boxCompare;
        while (boxes.size() != 0)
        {
            size_t index = 1;
            result.push_back(boxes[0]);
            while (boxes.size() > index)
            {
                boxMax.score = boxes[0].score;
                boxMax.classIndex = boxes[0].classIndex;
                boxMax.index = boxes[0].index;

                // translate point by maxLength * boxes[0].classIndex to
                // avoid bumping into two boxes of different classes
                boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
                boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
                boxMax.width = boxes[0].width;
                boxMax.height = boxes[0].height;

                boxCompare.score = boxes[index].score;
                boxCompare.classIndex = boxes[index].classIndex;
                boxCompare.index = boxes[index].index;

                // translate point by maxLength * boxes[0].classIndex to
                // avoid bumping into two boxes of different classes
                boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
                boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
                boxCompare.width = boxes[index].width;
                boxCompare.height = boxes[index].height;

                // the overlapping part of the two boxes
                float xLeft = max(boxMax.x, boxCompare.x);
                float yTop = max(boxMax.y, boxCompare.y);
                float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
                float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
                float width = max(0.0f, xRight - xLeft);
                float hight = max(0.0f, yBottom - yTop);
                float area = width * hight;
                float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

                // filter boxes by NMS threshold
                if (iou > NMSThreshold)
                {
                    boxes.erase(boxes.begin() + index);
                    continue;
                }
                ++index;
            }
            boxes.erase(boxes.begin());
        }

    }

    std::chrono::time_point<std::chrono::steady_clock> postEnd = std::chrono::steady_clock::now();
    postprocessMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    if (!enableDrawResult_)
    {
        saveImageMs = 0.0;
        (void)release;
        return SUCCESS;
    }

    std::chrono::time_point<std::chrono::steady_clock> visualizeStart = std::chrono::steady_clock::now();
    cv::Mat srcImage = cv::imread(imagePath);
    if (srcImage.empty())
    {
        ACLLITE_LOG_ERROR("read image failed: %s", imagePath.c_str());
        return FAILED;
    }
    double scaleX = srcWidth > 0 ? static_cast<double>(srcImage.cols) / static_cast<double>(srcWidth) : 1.0;
    double scaleY = srcHeight > 0 ? static_cast<double>(srcImage.rows) / static_cast<double>(srcHeight) : 1.0;

    // opencv draw label params
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255); // BGR
    const vector<cv::Scalar> colors{
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255)};

    int half = 2;
    size_t logCount = std::min(result.size(), maxLogDetPerImage_);
    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = static_cast<int>((result[i].x - result[i].width / half) * scaleX);
        leftUpPoint.y = static_cast<int>((result[i].y - result[i].height / half) * scaleY);
        rightBottomPoint.x = static_cast<int>((result[i].x + result[i].width / half) * scaleX);
        rightBottomPoint.y = static_cast<int>((result[i].y + result[i].height / half) * scaleY);
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        string className = (useBuiltinLabel && result[i].classIndex < builtinLabelNum)
                               ? label[result[i].classIndex]
                               : ("class_" + to_string(result[i].classIndex));
        string markString = to_string(result[i].score) + ":" + className;

        if (i < logCount)
        {
            ACLLITE_LOG_INFO("object detect [%s] success", markString.c_str());
        }

        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }
    if (result.size() > logCount)
    {
        ACLLITE_LOG_INFO("object detect log truncated: total=%ld, logged=%zu", result.size(), logCount);
    }
    std::chrono::time_point<std::chrono::steady_clock> visualizeEnd = std::chrono::steady_clock::now();
    visualizeMs = std::chrono::duration<double, std::milli>(visualizeEnd - visualizeStart).count();

    string savePath = "results/out_" + to_string(imageIndex) + ".jpg";
    std::chrono::time_point<std::chrono::steady_clock> saveStart = std::chrono::steady_clock::now();
    bool saveOk = cv::imwrite(savePath, srcImage);
    std::chrono::time_point<std::chrono::steady_clock> saveEnd = std::chrono::steady_clock::now();
    saveImageMs = std::chrono::duration<double, std::milli>(saveEnd - saveStart).count();
    if (!saveOk)
    {
        ACLLITE_LOG_ERROR("save image failed: %s", savePath.c_str());
        return FAILED;
    }
    (void)release;
    return SUCCESS;
}

void SampleYOLOV8::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main(int argc, char *argv[])
{
    const char *modelPath = "../model/yolov8s.om";
    string imagePath = "../data";
    int32_t modelWidth = 640;
    int32_t modelHeight = 640;
    bool enableDrawResult = true;
    size_t maxLogDetPerImage = 3;
    int32_t deviceId = 1;

    if (argc >= 2)
    {
        modelPath = argv[1];
    }
    if (argc >= 3)
    {
        modelWidth = std::atoi(argv[2]);
        modelHeight = modelWidth;
    }
    if (argc >= 4)
    {
        modelHeight = std::atoi(argv[3]);
    }
    if (argc >= 5)
    {
        imagePath = argv[4];
    }
    if (argc >= 6)
    {
        enableDrawResult = ParseBoolArg(argv[5], enableDrawResult);
    }
    if (argc >= 7)
    {
        int logCount = std::atoi(argv[6]);
        maxLogDetPerImage = (logCount > 0) ? static_cast<size_t>(logCount) : 0;
    }
    if (argc >= 8)
    {
        deviceId = std::atoi(argv[7]);
    }
    if (modelWidth <= 0 || modelHeight <= 0 ||
        (modelWidth % 32) != 0 || (modelHeight % 32) != 0)
    {
        ACLLITE_LOG_ERROR("invalid input size: %d x %d, width/height must be multiples of 32",
                          modelWidth, modelHeight);
        return FAILED;
    }
    ACLLITE_LOG_INFO("use model: %s, input size: %d x %d, image path: %s, draw=%d, maxLogDet=%zu, device=%d",
                     modelPath, modelWidth, modelHeight, imagePath.c_str(),
                     enableDrawResult ? 1 : 0, maxLogDetPerImage, deviceId);

    // all images in dir
    DIR *dir = opendir(imagePath.c_str());
    if (dir == nullptr)
    {
        ACLLITE_LOG_ERROR("file folder does no exist, please create folder %s", imagePath.c_str());
        return FAILED;
    }
    vector<string> allPath;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".keep") == 0)
        {
            continue;
        }
        else
        {
            string name = entry->d_name;
            string imgDir = imagePath + "/" + name;
            allPath.push_back(imgDir);
        }
    }
    closedir(dir);

    if (allPath.size() == 0)
    {
        ACLLITE_LOG_ERROR("the directory is empty, please download image to %s", imagePath.c_str());
        return FAILED;
    }

    // inference
    string fileName;
    bool release = false;
    SampleYOLOV8 sampleYOLO(modelPath, modelWidth, modelHeight, enableDrawResult, maxLogDetPerImage, deviceId);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }
    double preprocessSumMs = 0.0;
    double inferenceSumMs = 0.0;
    double d2hSumMs = 0.0;
    double postprocessSumMs = 0.0;
    double visualizeSumMs = 0.0;
    double saveImageSumMs = 0.0;
    size_t processedCount = 0;

    for (size_t i = 0; i < allPath.size(); i++)
    {
        if (allPath.size() == i)
        {
            release = true;
        }
        std::vector<InferenceOutput> inferOutputs;
        double preprocessMs = 0.0;
        double inferenceMs = 0.0;
        double d2hMs = 0.0;
        double postprocessMs = 0.0;
        double visualizeMs = 0.0;
        double saveImageMs = 0.0;
        fileName = allPath.at(i).c_str();
        std::chrono::time_point<std::chrono::steady_clock> t0 = std::chrono::steady_clock::now();
        ret = sampleYOLO.ProcessInput(fileName);
        std::chrono::time_point<std::chrono::steady_clock> t1 = std::chrono::steady_clock::now();
        preprocessMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            return FAILED;
        }
        
        std::chrono::time_point<std::chrono::steady_clock> t2 = std::chrono::steady_clock::now();
        ret = sampleYOLO.Inference(inferOutputs);
        std::chrono::time_point<std::chrono::steady_clock> t3 = std::chrono::steady_clock::now();
        inferenceMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }

        ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release, d2hMs, postprocessMs, visualizeMs, saveImageMs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
        double totalMs = preprocessMs + inferenceMs + d2hMs + postprocessMs + visualizeMs + saveImageMs;
        double fps = totalMs > 0.0 ? 1000.0 / totalMs : 0.0;
        ACLLITE_LOG_INFO("timing(ms): preprocess=%.3f, inference=%.3f, d2h=%.3f, postprocess=%.3f, visualize=%.3f, save=%.3f, total=%.3f, fps=%.3f",
                         preprocessMs, inferenceMs, d2hMs, postprocessMs, visualizeMs, saveImageMs, totalMs, fps);

        preprocessSumMs += preprocessMs;
        inferenceSumMs += inferenceMs;
        d2hSumMs += d2hMs;
        postprocessSumMs += postprocessMs;
        visualizeSumMs += visualizeMs;
        saveImageSumMs += saveImageMs;
        ++processedCount;
    }

    if (processedCount > 0)
    {
        ACLLITE_LOG_INFO("average timing(ms) for %zu images: preprocess=%.3f, inference=%.3f, d2h=%.3f, postprocess=%.3f, visualize=%.3f, save=%.3f",
                         processedCount,
                         preprocessSumMs / processedCount,
                         inferenceSumMs / processedCount,
                         d2hSumMs / processedCount,
                         postprocessSumMs / processedCount,
                         visualizeSumMs / processedCount,
                         saveImageSumMs / processedCount);
    }
    return SUCCESS;
}
