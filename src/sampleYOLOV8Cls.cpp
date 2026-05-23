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

// 分类结果结构
typedef struct ClassResult
{
    size_t classIndex;
    float score;
    string className;
} ClassResult;

class SampleYOLOV8Cls
{
public:
    SampleYOLOV8Cls(const char *modelPath, const char *yamlPath);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV8Cls();

private:
    Result LoadYamlConfig(const char *yamlPath);
    void ReleaseResource();

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
};

SampleYOLOV8Cls::SampleYOLOV8Cls(const char *modelPath, const char *yamlPath)
    : modelPath_(modelPath), modelWidth_(224), modelHeight_(224), classNum_(80)
{
    LoadYamlConfig(yamlPath);
}

SampleYOLOV8Cls::~SampleYOLOV8Cls()
{
    ReleaseResource();
}

Result SampleYOLOV8Cls::LoadYamlConfig(const char *yamlPath)
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

        return SUCCESS;
    }
    catch (const exception &e)
    {
        ACLLITE_LOG_ERROR("Failed to load yaml config: %s", e.what());
        return FAILED;
    }
}

Result SampleYOLOV8Cls::InitResource()
{
    // 初始化ACL资源
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

    // 初始化DVPP资源
    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // 加载模型
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8Cls::ProcessInput(string testImgPath)
{
    // 读取图像
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    // 拷贝图像到设备
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // JPEG解码为YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // 缩放到模型输入尺寸
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8Cls::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // 创建模型输入
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // 执行推理
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8Cls::GetResult(std::vector<InferenceOutput> &inferOutputs,
                                   string imagePath, size_t imageIndex, bool release)
{
    // YOLOv8分类输出: [1, num_classes]
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());

    // 找到最大概率的类别
    float maxScore = 0.0f;
    size_t maxIndex = 0;

    for (size_t i = 0; i < classNum_; ++i)
    {
        float score = classBuff[i];
        if (score > maxScore)
        {
            maxScore = score;
            maxIndex = i;
        }
    }

    // 获取类别名称
    string className = (maxIndex < classNames_.size()) ? classNames_[maxIndex] : to_string(maxIndex);

    ACLLITE_LOG_INFO("Classification result: class=%s, score=%.4f", className.c_str(), maxScore);

    // 在图像上绘制结果
    cv::Mat srcImage = cv::imread(imagePath);
    string resultText = className + ": " + to_string(maxScore);

    cv::putText(srcImage, resultText, cv::Point(10, 30),
                cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

    string savePath = "out_cls_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);

    if (release)
    {
        free(classBuff);
        classBuff = nullptr;
    }

    return SUCCESS;
}

void SampleYOLOV8Cls::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main(int argc, char *argv[])
{
    const char *modelPath = "../model/animals_cls_om/animals_cls_310P3_224.om";
    const char *yamlPath = "../model/animals_cls_om/dataset.yaml";
    const string imagePath = "../data";

    if (argc > 1)
    {
        modelPath = argv[1];
    }
    if (argc > 2)
    {
        yamlPath = argv[2];
    }

    // 读取所有图像
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

    // 推理
    SampleYOLOV8Cls sampleYOLOCls(modelPath, yamlPath);
    Result ret = sampleYOLOCls.InitResource();
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

        ret = sampleYOLOCls.ProcessInput(fileName);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            continue;
        }

        ret = sampleYOLOCls.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            continue;
        }

        ret = sampleYOLOCls.GetResult(inferOutputs, fileName, i, release);
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
