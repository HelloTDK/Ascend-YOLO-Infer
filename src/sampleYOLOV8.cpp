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
#include <cstdlib>
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

class SampleYOLOV8
{
public:
    SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight,
                 bool enableDrawResult, size_t maxLogDetPerImage);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release,
                     double &postprocessMs, double &saveImageMs);
    ~SampleYOLOV8();

private:
    size_t CalculateBoxNum(int32_t modelWidth, int32_t modelHeight) const;
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
    size_t modelOutputBoxNum_;
    size_t modelOutputChannelNum_;
    // 模型输出数据类型, 用于区分 fp16/fp32 读取
    aclDataType outputDataType_;
    bool enableDrawResult_;
    size_t maxLogDetPerImage_;
};

SampleYOLOV8::SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight,
                           bool enableDrawResult, size_t maxLogDetPerImage)
    : modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight), modelOutputBoxNum_(0),
      modelOutputChannelNum_(0), outputDataType_(ACL_FLOAT),
      enableDrawResult_(enableDrawResult), maxLogDetPerImage_(maxLogDetPerImage)
{
}

SampleYOLOV8::~SampleYOLOV8()
{
    ReleaseResource();
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

Result SampleYOLOV8::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8::GetResult(std::vector<InferenceOutput> &inferOutputs,
                               string imagePath, size_t imageIndex, bool release,
                               double &postprocessMs, double &saveImageMs)
{
    postprocessMs = 0.0;
    saveImageMs = 0.0;
    std::chrono::time_point<std::chrono::steady_clock> postStart = std::chrono::steady_clock::now();
    uint32_t outputDataBufId = 0;
    if (inferOutputs.size() <= outputDataBufId)
    {
        ACLLITE_LOG_ERROR("model output is empty");
        return FAILED;
    }

    // confidence threshold
    float confidenceThreshold = 0.35;

    size_t offset = 4;
    size_t outputChannelNum = modelOutputChannelNum_;
    size_t outputBoxNum = modelOutputBoxNum_;
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

    size_t classNum = outputChannelNum - offset;
    if (outputElementNum < (outputChannelNum * outputBoxNum))
    {
        ACLLITE_LOG_ERROR("output buffer is too small, elementNum=%zu, required=%zu",
                          outputElementNum, outputChannelNum * outputBoxNum);
        return FAILED;
    }

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

    const size_t builtinLabelNum = sizeof(label) / sizeof(label[0]);
    bool useBuiltinLabel = (classNum == builtinLabelNum);
    if (!useBuiltinLabel)
    {
        ACLLITE_LOG_WARNING("model class count %zu does not match label.h size %zu, use generic class names",
                            classNum, builtinLabelNum);
    }

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
    ACLLITE_LOG_INFO("decode layout selected: %s, inRangeRatio(CN)=%.4f, inRangeRatio(NC)=%.4f",
                     decodeChannelFirst ? "[C,N]" : "[N,C]",
                     channelFirstStat.first, channelLastStat.first);

    auto getOutputValue = [&](size_t boxIdx, size_t elemIdx) -> float
    {
        size_t flatIdx = decodeChannelFirst
                             ? (elemIdx * outputBoxNum + boxIdx)
                             : (boxIdx * outputChannelNum + elemIdx);
        return outputData[flatIdx];
    };

    int srcWidth = modelWidth_;
    int srcHeight = modelHeight_;
    cv::Mat srcImage;
    if (enableDrawResult_)
    {
        // draw mode only: load image for visualization
        srcImage = cv::imread(imagePath);
        if (srcImage.empty())
        {
            ACLLITE_LOG_ERROR("read image failed: %s", imagePath.c_str());
            return FAILED;
        }
        srcWidth = srcImage.cols;
        srcHeight = srcImage.rows;
    }

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

    ACLLITE_LOG_INFO("filter boxes by confidence threshold > %f success, boxes size is %ld", confidenceThreshold,boxes.size());

    // filter boxes by NMS
    vector<BoundBox> result;
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

    ACLLITE_LOG_INFO("filter boxes by NMS threshold > %f success, result size is %ld", NMSThreshold,result.size());
 
    // opencv draw label params
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255); // BGR
    const vector<cv::Scalar> colors{
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255)};

    if (!enableDrawResult_)
    {
        std::chrono::time_point<std::chrono::steady_clock> noDrawEnd = std::chrono::steady_clock::now();
        postprocessMs = std::chrono::duration<double, std::milli>(noDrawEnd - postStart).count();
        saveImageMs = 0.0;
        ACLLITE_LOG_INFO("postprocess(no-draw) result size is %ld", result.size());
        (void)release;
        return SUCCESS;
    }

    int half = 2;
    size_t logCount = std::min(result.size(), maxLogDetPerImage_);
    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / half;
        leftUpPoint.y = result[i].y - result[i].height / half;
        rightBottomPoint.x = result[i].x + result[i].width / half;
        rightBottomPoint.y = result[i].y + result[i].height / half;
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        string className = useBuiltinLabel ? label[result[i].classIndex] : ("class_" + to_string(result[i].classIndex));
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

    string savePath = "results/out_" + to_string(imageIndex) + ".jpg";
    std::chrono::time_point<std::chrono::steady_clock> saveStart = std::chrono::steady_clock::now();
    postprocessMs = std::chrono::duration<double, std::milli>(saveStart - postStart).count();
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
        enableDrawResult = (std::atoi(argv[5]) != 0);
    }
    if (argc >= 7)
    {
        int logCount = std::atoi(argv[6]);
        maxLogDetPerImage = (logCount > 0) ? static_cast<size_t>(logCount) : 0;
    }
    if (modelWidth <= 0 || modelHeight <= 0 ||
        (modelWidth % 32) != 0 || (modelHeight % 32) != 0)
    {
        ACLLITE_LOG_ERROR("invalid input size: %d x %d, width/height must be multiples of 32",
                          modelWidth, modelHeight);
        return FAILED;
    }
    ACLLITE_LOG_INFO("use model: %s, input size: %d x %d, image path: %s, draw=%d, maxLogDet=%zu",
                     modelPath, modelWidth, modelHeight, imagePath.c_str(),
                     enableDrawResult ? 1 : 0, maxLogDetPerImage);

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
    SampleYOLOV8 sampleYOLO(modelPath, modelWidth, modelHeight, enableDrawResult, maxLogDetPerImage);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }
    double preprocessSumMs = 0.0;
    double inferenceSumMs = 0.0;
    double postprocessSumMs = 0.0;
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
        double postprocessMs = 0.0;
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

        ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release, postprocessMs, saveImageMs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
        double totalMs = preprocessMs + inferenceMs + postprocessMs + saveImageMs;
        double fps = totalMs > 0.0 ? 1000.0 / totalMs : 0.0;
        ACLLITE_LOG_INFO("timing(ms): preprocess=%.3f, inference=%.3f, postprocess=%.3f, save=%.3f, total=%.3f, fps=%.3f",
                         preprocessMs, inferenceMs, postprocessMs, saveImageMs, totalMs, fps);

        preprocessSumMs += preprocessMs;
        inferenceSumMs += inferenceMs;
        postprocessSumMs += postprocessMs;
        saveImageSumMs += saveImageMs;
        ++processedCount;
    }

    if (processedCount > 0)
    {
        ACLLITE_LOG_INFO("average timing(ms) for %zu images: preprocess=%.3f, inference=%.3f, postprocess=%.3f, save=%.3f",
                         processedCount,
                         preprocessSumMs / processedCount,
                         inferenceSumMs / processedCount,
                         postprocessSumMs / processedCount,
                         saveImageSumMs / processedCount);
    }
    return SUCCESS;
}
