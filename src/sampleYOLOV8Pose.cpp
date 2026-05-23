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

typedef struct PoseKeyPoint
{
    float x;
    float y;
    float confidence;
} PoseKeyPoint;

typedef struct BoundBox
{
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
    vector<PoseKeyPoint> keypoints; // 关键点
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV8Pose
{
public:
    SampleYOLOV8Pose(const char *modelPath, const char *yamlPath);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV8Pose();

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
    size_t numKeypoints_;    // 关键点数量
    size_t keypointChannels_; // 每个关键点的通道数 (x, y, confidence)
    vector<string> classNames_;
    vector<string> keypointNames_;
    vector<cv::Scalar> colors_;
};

SampleYOLOV8Pose::SampleYOLOV8Pose(const char *modelPath, const char *yamlPath)
    : modelPath_(modelPath), modelWidth_(640), modelHeight_(640),
      classNum_(1), numKeypoints_(17), keypointChannels_(3)
{
    LoadYamlConfig(yamlPath);
}

SampleYOLOV8Pose::~SampleYOLOV8Pose()
{
    ReleaseResource();
}

size_t SampleYOLOV8Pose::CalculateBoxNum(int32_t imgSize)
{
    // 计算输出框数量: (imgSize/8)^2 + (imgSize/16)^2 + (imgSize/32)^2
    size_t s8 = (imgSize / 8) * (imgSize / 8);
    size_t s16 = (imgSize / 16) * (imgSize / 16);
    size_t s32 = (imgSize / 32) * (imgSize / 32);
    return s8 + s16 + s32;
}

Result SampleYOLOV8Pose::LoadYamlConfig(const char *yamlPath)
{
    try
    {
        YAML::Node config = YAML::LoadFile(yamlPath);

        // 读取图像尺寸
        if (config["imgsz"])
        {
            modelWidth_ = config["imgsz"].as<int>();
            modelHeight_ = modelWidth_;
            ACLLITE_LOG_INFO("Loaded imgsz: %d", modelWidth_);
        }

        // 读取类别名称
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

        // 读取关键点配置
        if (config["kpt_shape"])
        {
            YAML::Node kptShape = config["kpt_shape"];
            if (kptShape.size() >= 2)
            {
                numKeypoints_ = kptShape[0].as<int>();
                keypointChannels_ = kptShape[1].as<int>();
                ACLLITE_LOG_INFO("Loaded keypoint shape: [%zu, %zu]", numKeypoints_, keypointChannels_);
            }
        }

        // 读取关键点名称
        if (config["kpt_names"])
        {
            keypointNames_.clear();
            YAML::Node kptNames = config["kpt_names"];
            for (size_t i = 0; i < numKeypoints_; i++)
            {
                if (kptNames[i])
                {
                    keypointNames_.push_back(kptNames[i].as<string>());
                }
            }
        }

        // 读取颜色配置
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

        // 如果没有配置颜色，使用默认颜色
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

Result SampleYOLOV8Pose::InitResource()
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

Result SampleYOLOV8Pose::ProcessInput(string testImgPath)
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

Result SampleYOLOV8Pose::Inference(std::vector<InferenceOutput> &inferOutputs)
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

Result SampleYOLOV8Pose::GetResult(std::vector<InferenceOutput> &inferOutputs,
                                    string imagePath, size_t imageIndex, bool release)
{
    // YOLOv8-pose输出: [1, 56, 8400] (4 + 1 + 17*3)
    // 4: bbox (x, y, w, h)
    // 1: class confidence
    // 17*3: 17个关键点，每个3个值 (x, y, confidence)

    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());

    float confidenceThreshold = 0.35;
    size_t modelOutputBoxNum = CalculateBoxNum(modelWidth_);
    size_t offset = 4; // x, y, w, h

    cv::Mat srcImage = cv::imread(imagePath);
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    // 第一步：检测框筛选
    vector<BoundBox> boxes;
    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {
        // 对于pose，通常只有一个类别（人）
        float maxValue = 0;
        size_t maxIndex = 0;

        for (size_t j = 0; j < classNum_; ++j)
        {
            float value = classBuff[(offset + j) * modelOutputBoxNum + i];
            if (value > maxValue)
            {
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold)
        {
            BoundBox box;
            box.x = classBuff[i] * srcWidth / modelWidth_;
            box.y = classBuff[1 * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.width = classBuff[2 * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.height = classBuff[3 * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;

            // 提取关键点
            size_t kptOffset = offset + classNum_;
            for (size_t k = 0; k < numKeypoints_; ++k)
            {
                PoseKeyPoint kpt;
                kpt.x = classBuff[(kptOffset + k * keypointChannels_) * modelOutputBoxNum + i] * srcWidth / modelWidth_;
                kpt.y = classBuff[(kptOffset + k * keypointChannels_ + 1) * modelOutputBoxNum + i] * srcHeight / modelHeight_;
                kpt.confidence = classBuff[(kptOffset + k * keypointChannels_ + 2) * modelOutputBoxNum + i];
                box.keypoints.push_back(kpt);
            }

            boxes.push_back(box);
        }
    }

    ACLLITE_LOG_INFO("filter boxes by confidence threshold > %f success, boxes size is %ld",
                     confidenceThreshold, boxes.size());

    // 第二步：NMS
    vector<BoundBox> result;
    float NMSThreshold = 0.45;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);

    while (boxes.size() != 0)
    {
        result.push_back(boxes[0]);
        size_t index = 1;

        while (boxes.size() > index)
        {
            BoundBox boxMax = boxes[0];
            BoundBox boxCompare = boxes[index];

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
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }

    ACLLITE_LOG_INFO("filter boxes by NMS threshold > %f success, result size is %ld",
                     NMSThreshold, result.size());

    // 第三步：绘制结果
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255);

    // 定义关键点连接关系（骨架）
    vector<pair<int, int>> skeleton = {
        {0, 1}, {0, 2}, {1, 3}, {2, 3} // 根据实际关键点定义调整
    };

    for (size_t i = 0; i < result.size(); ++i)
    {
        // 绘制边界框
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / 2;
        leftUpPoint.y = result[i].y - result[i].height / 2;
        rightBottomPoint.x = result[i].x + result[i].width / 2;
        rightBottomPoint.y = result[i].y + result[i].height / 2;

        cv::Scalar color = colors_[i % colors_.size()];
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, color, lineSolid);

        // 绘制关键点
        for (size_t k = 0; k < result[i].keypoints.size(); ++k)
        {
            PoseKeyPoint kpt = result[i].keypoints[k];
            if (kpt.confidence > 0.5) // 只绘制置信度高的关键点
            {
                cv::circle(srcImage, cv::Point(kpt.x, kpt.y), 3, cv::Scalar(0, 255, 0), -1);
            }
        }

        // 绘制骨架连接
        for (const auto &conn : skeleton)
        {
            if (conn.first < result[i].keypoints.size() && conn.second < result[i].keypoints.size())
            {
                PoseKeyPoint kpt1 = result[i].keypoints[conn.first];
                PoseKeyPoint kpt2 = result[i].keypoints[conn.second];

                if (kpt1.confidence > 0.5 && kpt2.confidence > 0.5)
                {
                    cv::line(srcImage, cv::Point(kpt1.x, kpt1.y), cv::Point(kpt2.x, kpt2.y),
                             cv::Scalar(255, 0, 0), 2);
                }
            }
        }

        // 绘制标签
        string className = (result[i].classIndex < classNames_.size()) ? classNames_[result[i].classIndex] : to_string(result[i].classIndex);
        string markString = to_string(result[i].score) + ":" + className;

        ACLLITE_LOG_INFO("object pose detect [%s] success", markString.c_str());

        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }

    string savePath = "out_pose_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);

    if (release)
    {
        free(classBuff);
        classBuff = nullptr;
    }

    return SUCCESS;
}

void SampleYOLOV8Pose::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main(int argc, char *argv[])
{
    const char *modelPath = "../model/plate_pose_om/plate_pose_310P3_640.om";
    const char *yamlPath = "../model/plate_pose_om/dataset.yaml";
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

    SampleYOLOV8Pose sampleYOLOPose(modelPath, yamlPath);
    Result ret = sampleYOLOPose.InitResource();
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

        ret = sampleYOLOPose.ProcessInput(fileName);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            continue;
        }

        ret = sampleYOLOPose.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            continue;
        }

        ret = sampleYOLOPose.GetResult(inferOutputs, fileName, i, release);
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
