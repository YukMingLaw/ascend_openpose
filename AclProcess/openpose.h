#ifndef ACL_OPENPOSE_H
#define ACL_OPENPOSE_H

#include "iostream"
#include "opencv2/opencv.hpp"
#include "sys/time.h"

using namespace std;
using namespace cv;

#define POSE_COCO_COLORS_RENDER_GPU \
	255.f,     0.f,    85.f, \
	170.f,     0.f,   255.f, \
	255.f,     0.f,   170.f, \
	85.f,     0.f,   255.f, \
	255.f,     0.f,   255.f, \
	170.f,   255.f,     0.f, \
	255.f,    85.f,     0.f, \
	85.f,   255.f,     0.f, \
	255.f,   170.f,     0.f, \
	0.f,   255.f,     0.f, \
	255.f,   255.f,     0.f, \
	0.f,   170.f,   255.f, \
	0.f,   255.f,    85.f, \
	0.f,    85.f,   255.f, \
	0.f,   255.f,   170.f, \
	0.f,     0.f,   255.f, \
	0.f,   255.f,   255.f, \
	255.f,     0.f,     0.f, \
	255.f,     0.f,     0.f, \
	0.f,     0.f,   255.f, \
	0.f,     0.f,   255.f, \
	0.f,     0.f,   255.f, \
	0.f,   255.f,   255.f, \
	0.f,   255.f,   255.f, \
	0.f,   255.f,   255.f

template<typename T>
inline int intRound(const T a)
{
    return int(a+0.5f);
}
template<typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

struct BlobData{
    int count;
    float* list;
    int num;
    int channels;
    int height;
    int width;
    int capacity_count;
};


class OpenPose{
public:
    void init(vector<int>& _modelInputShape,
              int _resizeScalar,
              float _nmsThreshold = 0.2, // 0.05
              float _interMinAboveThreshold = 0.7, // 0.95
              float _interThreshold = 0.2, //0.05
              int _minSubsetCnt = 3,
              float _minSubsetScore = 0.4);
    void deInit();
    void postprocess(Mat& img, float* netOutputPtr);
	Mat preprocess(Mat& img);
private:
    std::vector<float> POSE_COCO_COLORS_RENDER{ POSE_COCO_COLORS_RENDER_GPU };
    std::vector<unsigned int> POSE_COCO_PAIRS_RENDER{1,8, 1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 8,9, 9,10, 10,11, 8,12, 12,13, 13,14, 1,0, 0,15, 15,17, 0,16, 16,18, 14,19, 19,20, 14,21, 11,22, 22,23, 11,24};
    int POSE_MAX_PEOPLE = 128;
    vector<int> modelInputShape;
    int resizeScalar;
    float nmsThreshold;
    float interMinAboveThreshold;
    float interThreshold;
    int minSubsetCnt;
    float minSubsetScore;
    float scale;
    BlobData* peaks;
    BlobData* net_output;
    BlobData* heapmaps;
    Size baseSize;

    BlobData* createBlob_local(int num, int channels, int height, int width);
    void releaseBlob_local(BlobData** blob);
	float resizeGetScaleFactor(Size originSize, Size modelInputSize);
	Mat resizeFixedAspectRatio(cv::Mat& cvMat, Size modelInputSize);
    void nms(BlobData* bottom_blob, BlobData* top_blob, float threshold);
    void connectBodyPartsCpu(vector<float>& poseKeypoints, float* heatMapPtr, float* peaksPtr,
                                       Size& heatMapSize, int maxPeaks, float interMinAboveThreshold,
                                       float interThreshold, int minSubsetCnt, float minSubsetScore, float scaleFactor, vector<int>& keypointShape);
    void renderKeypointsCpu(Mat& frame, vector<float>& keypoints, vector<int> keyshape, std::vector<unsigned int>& pairs,
                                      std::vector<float> colors, float thicknessCircleRatio, float thicknessLineRatioWRTCircle,
                                      float threshold, float scale);
    void renderPoseKeypointsCpu(Mat& frame, vector<float>& poseKeypoints, vector<int> keyshape,
                                 float renderThreshold, float scale, bool blendOriginalFrame = true);


};

#endif //ACL_OPENPOSE_H
