#include <iostream>
#include "AclProcess.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc,char** argv)
{
    VideoCapture cap;
    cap.open(argv[2]);
    cv::VideoWriter video;
    video.open( "result.avi", 
                cv::VideoWriter::fourcc('H', '2', '6', '4'), 
                60.0,
                cv::Size(
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), 
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                    )
                );
    AclProcess aclprocess;
    aclError ret = aclprocess.Init(0, argv[1]);
    if(ret != ACL_ERROR_NONE){
        cout << "AclProcess Init faild." << endl;
        return -1;
    }
    Mat img;
    while(cap.read(img)){
        aclprocess.Process(img);
        video.write(img);
//        waitKey(1);
    }

    return 0;
}
