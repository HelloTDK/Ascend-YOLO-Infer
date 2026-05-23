#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include <chrono>

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
    float angle;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV8OBB
{
public:
    SampleYOLOV8OBB(const char *modelPath, const char *yamlPath);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV8OBB();

private:
    Result LoadYamlConfig(const char *yamlPath);
    void ReleaseResource();
    size_t CalculateBoxNum(int32_t imgSize);

    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;

    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
    size_t classNum_;
    vector<string> classNames_;
    vector<cv::Scalar> colors_;
};

SampleYOLOV8OBB::SampleYOLOV8OBB(const char *modelPath, const char *yamlPath)
    : modelPath_(modelPath), modelWidth_(1024), modelHeight_(1024), classNum_(1)
{
    LoadYamlConfig(yamlPath);
}

SampleYOLOV8OBB::~SampleYOLOV8OBB()
{
    ReleaseResource();
}

size_t SampleYOLOV8OBB::CalculateBoxNum(int32_t imgSize)
{
    size_t s8 = (imgSize / 8) * (imgSize / 8);
    size_t s16 = (imgSize / 16) * (imgSize / 16);
    size_t s32 = (imgSize / 32) * (imgSize / 32);
    return s8 + s16 + s32;
}

Result SampleYOLOV8OBB::LoadYamlConfig(const char *yamlPath)
{
    try
    {
        YAML::Node config = YAML::LoadFile(yamlPath);

        if (config["imgsz"])
        {
            modelWidth_ = config["imgsz"].as<int>();
            modelHeight_ = modelWidth_;
            ACLLITE_LOG_INFO("Loaded imgsz: %d", modelWidth_);
        }

        if (config["names"])
        {
            classNames_.clear();
            YAML::Node names = config["names"];
            classNum_ = names.size();

            for (size_t i = 0; i < classNum_; i++)
            {
                if (names[i])
                {
                    classNames_.push_back(names[i].as<string>());
                }
            }
            ACLLITE_LOG_INFO("Loaded %zu classes", classNum_);
        }

        if (config["colors"])
        {
            colors_.clear();
            YAML::Node colors = config["colors"];
            for (size_t i = 0; i < colors.size(); i++)
            {
                if (colors[i].size() >= 3)
                {
                    int b = colors[i][0].as<int>();
                    int g = colors[i][1].as<int>();
                    int r = colors[i][2].as<int>();
                    colors_.push_back(cv::Scalar(b, g, r));
                }
            }
        }

        if (colors_.empty())
        {
            colors_.push_back(cv::Scalar(255, 0, 0));
            colors_.push_back(cv::Scalar(0, 255, 0));
            colors_.push_back(cv::Scalar(0, 0, 255));
        }

        return SUCCESS;
    }
    catch (const exception &e)
    {
        ACLLITE_LOG_ERROR("Failed to load yaml config: %s", e.what());
        return FAILED;
    }
}

Result SampleYOLOV8OBB::InitResource()
{
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

    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8OBB::ProcessInput(string testImgPath)
{
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8OBB::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8OBB::GetResult(std::vector<InferenceOutput> &inferOutputs,
                                  string imagePath, size_t imageIndex, bool release)
{
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());

    ACLLITE_LOG_INFO("Output buffer size: %ld bytes", inferOutputs[outputDataBufId].size);
    ACLLITE_LOG_INFO("First 20 values: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
        classBuff[0], classBuff[1], classBuff[2], classBuff[3], classBuff[4],
        classBuff[5], classBuff[6], classBuff[7], classBuff[8], classBuff[9],
        classBuff[10], classBuff[11], classBuff[12], classBuff[13], classBuff[14],
        classBuff[15], classBuff[16], classBuff[17], classBuff[18], classBuff[19]);

    float confidenceThreshold = 0.8;
    size_t modelOutputBoxNum = CalculateBoxNum(modelWidth_);

    ACLLITE_LOG_INFO("Model output box num: %ld", modelOutputBoxNum);

    cv::Mat srcImage = cv::imread(imagePath);
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    vector<BoundBox> boxes;
    size_t offset = 5;

    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {
        float maxValue = 0;
        size_t maxIndex = 0;

        for (size_t j = 0; j < classNum_; ++j)
        {
            float logit = classBuff[(5 + j) * modelOutputBoxNum + i];
            float value = 1.0f / (1.0f + exp(-logit));
            if (value > maxValue)
            {
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold)
        {
            BoundBox box;
            box.x = classBuff[0 * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.y = classBuff[1 * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.width = classBuff[2 * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.height = classBuff[3 * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.angle = classBuff[4 * modelOutputBoxNum + i];
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;

            boxes.push_back(box);
        }
    }

    ACLLITE_LOG_INFO("filter boxes by confidence threshold > %f success, boxes size is %ld",
                     confidenceThreshold, boxes.size());

    vector<BoundBox> result;
    float NMSThreshold = 0.45;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);

    vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (suppressed[j]) continue;

            BoundBox boxMax = boxes[i];
            BoundBox boxCompare = boxes[j];

            boxMax.x += maxLength * boxMax.classIndex;
            boxMax.y += maxLength * boxMax.classIndex;
            boxCompare.x += boxCompare.classIndex * maxLength;
            boxCompare.y += boxCompare.classIndex * maxLength;

            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float height = max(0.0f, yBottom - yTop);
            float area = width * height;
            float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            if (iou > NMSThreshold)
            {
                suppressed[j] = true;
            }
        }
    }

    ACLLITE_LOG_INFO("filter boxes by NMS threshold > %f success, result size is %ld",
                     NMSThreshold, result.size());

    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255);

    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Point2f center(result[i].x, result[i].y);
        cv::Size2f size(result[i].width, result[i].height);
        float angle = result[i].angle;

        cv::RotatedRect rotRect(center, size, angle);
        cv::Point2f vertices[4];
        rotRect.points(vertices);

        cv::Scalar color = colors_[i % colors_.size()];
        for (int j = 0; j < 4; j++)
        {
            cv::line(srcImage, vertices[j], vertices[(j + 1) % 4], color, lineSolid);
        }

        string className = (result[i].classIndex < classNames_.size()) ? classNames_[result[i].classIndex] : to_string(result[i].classIndex);
        string markString = to_string(result[i].score) + ":" + className;

        ACLLITE_LOG_INFO("object OBB detect [%s] success", markString.c_str());

        cv::putText(srcImage, markString, cv::Point(result[i].x - result[i].width / 2, result[i].y - result[i].height / 2 + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }

    string savePath = "out_obb_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);

    return SUCCESS;
}

void SampleYOLOV8OBB::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main(int argc, char *argv[])
{
    const char *modelPath = "../model/person_obb_om/person_obb_310P3_1024.om";
    const char *yamlPath = "../model/person_obb_om/dataset.yaml";
    const string imagePath = "../data";

    if (argc > 1)
    {
        modelPath = argv[1];
    }
    if (argc > 2)
    {
        yamlPath = argv[2];
    }

    DIR *dir = opendir(imagePath.c_str());
    if (dir == nullptr)
    {
        ACLLITE_LOG_ERROR("file folder does not exist, please create folder %s", imagePath.c_str());
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

    SampleYOLOV8OBB sampleYOLOOBB(modelPath, yamlPath);
    Result ret = sampleYOLOOBB.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    for (size_t i = 0; i < allPath.size(); i++)
    {
        bool release = (i == allPath.size() - 1);
        std::vector<InferenceOutput> inferOutputs;
        string fileName = allPath.at(i);

        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

        ret = sampleYOLOOBB.ProcessInput(fileName);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            continue;
        }

        ret = sampleYOLOOBB.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            continue;
        }

        ret = sampleYOLOOBB.GetResult(inferOutputs, fileName, i, release);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            continue;
        }

        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        ACLLITE_LOG_INFO("Inference elapsed time: %f s, fps is %f", elapsed.count(), 1 / elapsed.count());
    }

    return SUCCESS;
}
