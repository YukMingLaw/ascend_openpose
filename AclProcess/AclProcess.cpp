#include "AclProcess.h"

AclProcess::AclProcess()
{

}
AclProcess::~AclProcess()
{
    openpose.deInit();
    m_modelProcess = nullptr;
    aclError ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "some tasks in stream not done, ret = " << ret <<endl;
    }
    cout << "all tasks in stream done" << endl;
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Stream faild, ret = " << ret <<endl;
    }
    cout << "Destroy Stream successfully" << endl;
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Context faild, ret = " << ret <<endl;
    }
    cout << "Destroy Context successfully" << endl;
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to deinit acl, ret = " << ret <<endl;
    }
    cout << "acl deinit successfully" << endl;

}

int AclProcess::Init(int deviceId, string modelPath)
{
    //Init
    aclError ret = aclInit(nullptr); // Initialize ACL
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to init acl, ret = " << ret <<endl;
        return ret;
    }
    cout << "acl init successfully" << endl;
    ret = aclrtCreateContext(&context_, deviceId);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    cout << "Create context successfully" << endl;
    ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    cout << "set context successfully" << endl;
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to create stream, ret = " << ret << endl;
        return ret;
    }
    cout << "Create stream successfully" << endl;
    //Load model
    if (m_modelProcess == nullptr) {
        m_modelProcess = std::make_shared<ModelProcess>(deviceId, "");
    }
    ret = m_modelProcess->Init(modelPath);

    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to initialize m_modelProcess, ret = " << ret << endl;
        return ret;
    }
    m_modelDesc = m_modelProcess->GetModelDesc();
    //get model input description and malloc them
    size_t inputSize = aclmdlGetNumInputs(m_modelDesc);
    for (size_t i = 0; i < inputSize; i++) {
        size_t bufferSize = aclmdlGetInputSizeByIndex(m_modelDesc, i);
        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        inputBuffers.push_back(inputBuffer);
        inputSizes.push_back(bufferSize);
    }
    //get model output description and malloc them
    size_t outputSize = aclmdlGetNumOutputs(m_modelDesc);
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(m_modelDesc, i);
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    vector<int> modelInputShape(4);
    aclmdlIODims dims;
    aclmdlGetInputDims(m_modelDesc,0,&dims);
    if(dims.dimCount == 4){
        modelInputShape[0] = dims.dims[0];
        modelInputShape[2] = dims.dims[1];
        modelInputShape[3] = dims.dims[2];
        modelInputShape[1] = dims.dims[3];
    }
    openpose.init(modelInputShape, 8);
    cout << "finish init AclProcess" << endl;
    return ACL_ERROR_NONE;
}

void AclProcess::Process(Mat& img)
{
    aclError ret = ACL_ERROR_NONE;
    struct timeval begin;
    struct timeval end;
    gettimeofday(&begin,nullptr);
    cv::Mat resize_img = openpose.preprocess(img);
    aclrtMemcpy(inputBuffers[0], inputSizes[0], resize_img.data, resize_img.cols * resize_img.rows * resize_img.channels(), ACL_MEMCPY_HOST_TO_DEVICE);
    //forward
    ret = m_modelProcess->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != ACL_ERROR_NONE) {
        cout<<"model run faild.ret = "<< ret <<endl;
    }
    //postprocess
    void* newresult;
    ret = (aclError)aclrtMallocHost(&newresult, outputSizes[0]);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to malloc output buffer of model on host, ret = " << ret << endl;
    }
    ret = (aclError)aclrtMemcpy(newresult, outputSizes[0], outputBuffers[0], outputSizes[0], ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to copy output buffer of model from device to host, ret = " << ret << endl;
    }
    openpose.postprocess(img, (float*)newresult);
    aclrtFreeHost(newresult);
    gettimeofday(&end,nullptr);
    cout<<"fps:"<<1000.0 / ((end.tv_sec-begin.tv_sec)*1000+(end.tv_usec-begin.tv_usec) / 1000.0)<<endl;
    cout << "=======================" <<endl;
}
