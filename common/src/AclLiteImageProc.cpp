/**
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File AclLiteImageProc.cpp
* Description: handle dvpp process
*/
#include <iostream>
#include <unordered_map>
#include "acl/acl.h"
#include "ResizeHelper.h"
#include "JpegDHelper.h"
#include "JpegEHelper.h"
#include "PngDHelper.h"
#include "AclLiteImageProc.h"
#include "CropAndPasteHelper.h"

using namespace std;

static std::unordered_map<string, acldvppChannelMode> STR2MODE = {
        {"DVPP_CHNMODE_VPC", DVPP_CHNMODE_VPC},
        {"DVPP_CHNMODE_JPEGD", DVPP_CHNMODE_JPEGD},
        {"DVPP_CHNMODE_JPEGE", DVPP_CHNMODE_JPEGE},
        {"DVPP_CHNMODE_PNGD", DVPP_CHNMODE_PNGD}
};

AclLiteImageProc::AclLiteImageProc():isReleased_(false), stream_(nullptr),
    dvppChannelDesc_(nullptr), isInitOk_(false)
{
}

AclLiteImageProc::~AclLiteImageProc()
{
    DestroyResource();
}
// 销毁资源
void AclLiteImageProc::DestroyResource()
{
    if (isReleased_) {
        return;
    }

    aclError aclRet;

    if (dvppChannelDesc_ != nullptr) {
        aclRet = acldvppDestroyChannel(dvppChannelDesc_);   // 销毁dvpp通道
        if (aclRet != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Destroy dvpp channel error: %d", aclRet);
        }

        (void)acldvppDestroyChannelDesc(dvppChannelDesc_);  // 销毁dvpp通道描述符
        dvppChannelDesc_ = nullptr;
    }

    if (stream_ != nullptr) {
        aclRet = aclrtDestroyStream(stream_);       // 销毁stream
        if (aclRet != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Vdec destroy stream failed, error %d", aclRet);
        }
        stream_ = nullptr;
    }

    isReleased_ = true;
}
// 初始化dvpp资源
AclLiteError AclLiteImageProc::Init(string mode)
{
    aclError aclRet = aclrtCreateStream(&stream_);  // 创建stream
    if (aclRet != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("Create venc stream failed, error %d", aclRet);
        return ACLLITE_ERROR_CREATE_STREAM;
    }

    dvppChannelDesc_ = acldvppCreateChannelDesc();  // 创建dvpp通道描述符
    if (dvppChannelDesc_ == nullptr) {
        ACLLITE_LOG_ERROR("Create dvpp channel desc failed");
        return ACLLITE_ERROR_CREATE_DVPP_CHANNEL_DESC;
    }

    auto socVersion = aclrtGetSocName();            // 获取soc版本
    if (strncmp(socVersion, "Ascend310P3", sizeof("Ascend310P3") - 1) == 0 && mode != "") {
        aclRet = acldvppSetChannelDescMode(dvppChannelDesc_, STR2MODE[mode]);
        if (aclRet != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("acldvppCreateChannel failed, aclRet = %d", aclRet);
            return ACLLITE_ERRROR_CREATE_DVPP_CHANNEL;
        }
    }

    aclRet = acldvppCreateChannel(dvppChannelDesc_);    // 创建dvpp通道
    if (aclRet != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("acldvppCreateChannel failed, aclRet = %d", aclRet);
        return ACLLITE_ERRROR_CREATE_DVPP_CHANNEL;
    }

    isInitOk_ = true;
    ACLLITE_LOG_INFO("dvpp init resource ok");

    return ACLLITE_OK;
}
// 图像缩放
AclLiteError AclLiteImageProc::Resize(ImageData& dest, ImageData& src,
                                      uint32_t width, uint32_t height)
{
    ResizeHelper resizeOp(stream_, dvppChannelDesc_, width, height);
    return resizeOp.Process(dest, src); // 调用ResizeHelper进行图像缩放
}
// JPEG解码
AclLiteError AclLiteImageProc::JpegD(ImageData& dest, ImageData& src)
{
    JpegDHelper jpegD(stream_, dvppChannelDesc_);
    return jpegD.Process(dest, src);    // 调用JpegDHelper进行JPEG解码
}
// PNG解码
AclLiteError AclLiteImageProc::PngD(ImageData& dest, ImageData& src)
{
    PngDHelper PngD(stream_, dvppChannelDesc_);
    return PngD.Process(dest, src);     // 调用PngDHelper进行PNG解码
}

// 裁剪  CropAndPasteHelper   
AclLiteError AclLiteImageProc::Crop(ImageData& dest, ImageData& src,
                                    uint32_t ltHorz, uint32_t ltVert,
                                    uint32_t rbHorz, uint32_t rbVert)
{
    CropAndPasteHelper crop(stream_, dvppChannelDesc_,
                            ltHorz, ltVert, rbHorz, rbVert);
    return crop.Process(dest, src);     // 调用CropAndPasteHelper进行图像裁剪
}
// 图像粘贴
AclLiteError AclLiteImageProc::CropPaste(ImageData& dest, ImageData& src,
                                         uint32_t width, uint32_t height,
                                         uint32_t ltHorz, uint32_t ltVert,
                                         uint32_t rbHorz, uint32_t rbVert)
{
    CropAndPasteHelper crop(stream_, dvppChannelDesc_,
                            width, height, ltHorz,
                            ltVert, rbHorz, rbVert);
    return crop.ProcessCropPaste(dest, src);    // 调用CropAndPasteHelper进行图像粘贴
}
// 图像等比例粘贴
AclLiteError AclLiteImageProc::ProportionPaste(ImageData& dest, ImageData& src,
                                               uint32_t ltHorz, uint32_t ltVert,
                                               uint32_t rbHorz, uint32_t rbVert)
{
    CropAndPasteHelper crop(stream_, dvppChannelDesc_,
                            ltHorz, ltVert, rbHorz, rbVert);
    return crop.ProportionProcess(dest, src);   // 调用CropAndPasteHelper进行图像等比例粘贴
}
// 图像等比例居中粘贴
AclLiteError AclLiteImageProc::ProportionPasteCenter(ImageData& dest, ImageData& src,
                                                     uint32_t width, uint32_t height)
{
    CropAndPasteHelper crop(stream_, dvppChannelDesc_,
                            0, 0, width, height);
    return crop.ProportionCenterProcess(dest, src); // 调用CropAndPasteHelper进行图像等比例居中粘贴
}
// JPEG编码
AclLiteError AclLiteImageProc::JpegE(ImageData& dest, ImageData& src)
{
    JpegEHelper jpegE(stream_, dvppChannelDesc_);
    return jpegE.Process(dest, src);    // 调用JpegEHelper进行JPEG编码
}