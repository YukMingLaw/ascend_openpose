#pragma once

#include "iostream"
#include "acl/acl.h"
#include "ModelProcess.h"
#include "opencv2/opencv.hpp"
#include "sys/time.h"
#include "openpose.h"

using namespace std;
using namespace cv;

struct ObjDetectInfo {
    float leftTopX;
    float leftTopY;
    float rightBotX;
    float rightBotY;
    float confidence;
    float classId;
};

class AclProcess{
public:
    AclProcess();
    ~AclProcess();
    int Init(int deviceId, string modelPath);
    void Process(Mat& img);
private:
    OpenPose openpose;
    std::vector<void *> inputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<ModelProcess> m_modelProcess;
    aclmdlDesc *m_modelDesc;
};
