#include "openpose.h"
//#define PRINT_TIME

float OpenPose::resizeGetScaleFactor(Size originSize, Size modelInputSize)
{
    const auto ratioWidth = (modelInputSize.width - 1) / (float)(originSize.width - 1);
    const auto ratioHeight = (modelInputSize.height - 1) / (float)(originSize.height - 1);
    return ratioWidth > ratioHeight ? ratioHeight : ratioWidth;
}

Mat OpenPose::resizeFixedAspectRatio(cv::Mat& cvMat, Size modelInputSize)
{
    cv::Mat resizedCvMat;
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    float scaleFactor = resizeGetScaleFactor(Size(cvMat.cols, cvMat.rows), modelInputSize);
    scale = 1 / scaleFactor;
    M.at<double>(0,0) = scaleFactor;
    M.at<double>(1,1) = scaleFactor;
    if (scaleFactor != 1. || modelInputSize != cvMat.size()){
        cv::warpAffine(cvMat, resizedCvMat, M, modelInputSize, (scaleFactor > 1. ? cv::INTER_CUBIC : cv::INTER_AREA), cv::BORDER_CONSTANT, cv::Scalar{0,0,0});
    }
    else{
        cvMat.copyTo(resizedCvMat);
    }
    return resizedCvMat;
}

void OpenPose::connectBodyPartsCpu(vector<float>& poseKeypoints, float* heatMapPtr, float* peaksPtr,
                                   Size& heatMapSize, int maxPeaks, float interMinAboveThreshold,
                                   float interThreshold, int minSubsetCnt, float minSubsetScore, float scaleFactor, vector<int>& keypointShape)
{
    keypointShape.resize(3);

    const std::vector<unsigned int> POSE_COCO_PAIRS{1,8,  1,2,   1,5,   2,3,   3,4,   5,6,   6,7,  8,9, 9,10, 10,11, 8,12, 12,13, 13,14, 1,0,   0,15,  15,17, 0,16,  16,18, 14,19, 19,20, 14,21, 11,22, 22,23, 11,24};
    const std::vector<unsigned int> POSE_COCO_MAP_IDX{0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3,   4,5,   8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 40,41, 42,43, 44,45, 46,47, 48,49, 50,51};
    const auto& bodyPartPairs = POSE_COCO_PAIRS;
    const auto& mapIdx = POSE_COCO_MAP_IDX;
    const auto numberBodyParts = 25;

    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

    std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
    const auto subsetCounterIndex = numberBodyParts;
    const auto subsetSize = numberBodyParts + 1;

    const auto peaksOffset = 3 * (maxPeaks + 1);
    const auto heatMapOffset = heatMapSize.area();

    for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
    {
        const auto bodyPartA = bodyPartPairs[2 * pairIndex];
        const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
        const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
        const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
        const auto nA = intRound(candidateA[0]);
        const auto nB = intRound(candidateB[0]);

        // add parts into the subset in special case
        if (nA == 0 || nB == 0)
        {
            // Change w.r.t. other
            if (nA == 0) // nB == 0 or not
            {
                for (auto i = 1; i <= nB; i++)
                {
                    bool num = false;
                    const auto indexB = bodyPartB;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
                        if (subset[j].first[indexB] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
            else // if (nA != 0 && nB == 0)
            {
                for (auto i = 1; i <= nA; i++)
                {
                    bool num = false;
                    const auto indexA = bodyPartA;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
                        if (subset[j].first[indexA] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
        }
        else // if (nA != 0 && nB != 0)
        {
            std::vector<std::tuple<double, int, int>> temp;

            const auto* const mapX = heatMapPtr + (mapIdx[2 * pairIndex] + 26) * heatMapOffset;
            const auto* const mapY = heatMapPtr + (mapIdx[2 * pairIndex + 1] + 26) * heatMapOffset;
            for (auto i = 1; i <= nA; i++)
            {
                for (auto j = 1; j <= nB; j++)
                {
                    const auto dX = candidateB[j * 3] - candidateA[i * 3];
                    const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
                    const auto vectorAToBMax = fastMax(std::abs(dX), std::abs(dY));
                    const auto numInter = fastMax(5, fastMin(25, intRound(std::sqrt(5 * vectorAToBMax))));
                    const auto normVec = float(std::sqrt(dX*dX + dY*dY));
                    // If the peaksPtr are coincident. Don't connect them.
                    if (normVec > 1e-6)
                    {
                        const auto sX = candidateA[i * 3];
                        const auto sY = candidateA[i * 3 + 1];
                        const auto vecX = dX / normVec;
                        const auto vecY = dY / normVec;

                        auto sum = 0.;
                        auto count = 0;
                        for (auto lm = 0; lm < numInter; lm++)
                        {
                            const auto mX = fastMax(0, fastMin(heatMapSize.width - 1, intRound(sX + lm * dX / numInter)));
                            const auto mY = fastMax(0, fastMin(heatMapSize.height - 1, intRound(sY + lm * dY / numInter)));
                            //checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
                            //checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
                            const auto idx = mY * heatMapSize.width + mX;
                            const auto score = (vecX * mapX[idx] + vecY * mapY[idx]);
                            if (score > interThreshold)
                            {
                                sum += score;
                                count++;
                            }
                        }

                        // parts score + connection score
                        if (count / numInter  > interMinAboveThreshold){
                            temp.emplace_back(std::make_tuple(sum / count, i, j));
                        }
                    }
                }
            }

            // select the top minAB connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (!temp.empty())
                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

            std::vector<std::tuple<int, int, double>> connectionK;

            const auto minAB = fastMin(nA, nB);
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);
            auto counter = 0;
            for (auto row = 0u; row < temp.size(); row++)
            {
                const auto score = std::get<0>(temp[row]);
                const auto x = std::get<1>(temp[row]);
                const auto y = std::get<2>(temp[row]);
                if (!occurA[x - 1] && !occurB[y - 1])
                {
                    connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
                                                             bodyPartB*peaksOffset + y * 3 + 2,
                                                             score));
                    counter++;
                    if (counter == minAB)
                        break;
                    occurA[x - 1] = 1;
                    occurB[y - 1] = 1;
                }
            }

            // Cluster all the body part candidates into subset based on the part connection
            // initialize first body part connection 15&16
            if (pairIndex == 0)
            {
                for (const auto connectionKI : connectionK)
                {
                    std::vector<int> rowVector(numberBodyParts + 3, 0);
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    const auto score = std::get<2>(connectionKI);
                    rowVector[bodyPartPairs[0]] = indexA;
                    rowVector[bodyPartPairs[1]] = indexB;
                    rowVector[subsetCounterIndex] = 2;
                    // add the score of parts and the connection
                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                }
            }
                // Add ears connections (in case person is looking to opposite direction to camera)
            else if ((numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                     || ((numberBodyParts == 19 || (numberBodyParts == 25)
                          || numberBodyParts == 59 || numberBodyParts == 65)
                         && (pairIndex==18 || pairIndex==19))
                    )//(pairIndex == 17 || pairIndex == 18)
            {
                for (const auto& connectionKI : connectionK)
                {
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    for (auto& subsetJ : subset)
                    {
                        auto& subsetJFirst = subsetJ.first[bodyPartA];
                        auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
                        if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
                            subsetJFirstPlus1 = indexB;
                        else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
                            subsetJFirst = indexA;
                    }
                }
            }
            else
            {
                if (!connectionK.empty())
                {
                    // A is already in the subset, find its connection B
                    for (auto i = 0u; i < connectionK.size(); i++)
                    {
                        const auto indexA = std::get<0>(connectionK[i]);
                        const auto indexB = std::get<1>(connectionK[i]);
                        const auto score = std::get<2>(connectionK[i]);
                        auto num = 0;
                        for (auto j = 0u; j < subset.size(); j++)
                        {
                            if (subset[j].first[bodyPartA] == indexA)
                            {
                                subset[j].first[bodyPartB] = indexB;
                                num++;
                                subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
                                subset[j].second = subset[j].second + peaksPtr[indexB] + score;
                            }
                        }
                        // if can not find partA in the subset, create a new subset
                        if (num == 0)
                        {
                            std::vector<int> rowVector(subsetSize, 0);
                            rowVector[bodyPartA] = indexA;
                            rowVector[bodyPartB] = indexB;
                            rowVector[subsetCounterIndex] = 2;
                            const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                            subset.emplace_back(std::make_pair(rowVector, subsetScore));
                        }
                    }
                }
            }
        }
    }

    auto numberPeople = 0;
    std::vector<int> validSubsetIndexes;
    validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
    for (auto index = 0u; index < subset.size(); index++)
    {
        const auto subsetCounter = subset[index].first[subsetCounterIndex];
        const auto subsetScore = subset[index].second;
        if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
        {
            numberPeople++;
            validSubsetIndexes.emplace_back(index);
            if (numberPeople == POSE_MAX_PEOPLE)
                break;
        }
        else if (subsetCounter < 1)
            printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
    }

    // Fill and return poseKeypoints
    keypointShape = { numberPeople, (int)numberBodyParts, 3 };
    if (numberPeople > 0)
        poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
    else
        poseKeypoints.clear();

    for (auto person = 0u; person < validSubsetIndexes.size(); person++)
    {
        const auto& subsetI = subset[validSubsetIndexes[person]].first;
        for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
        {
            const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
            const auto bodyPartIndex = subsetI[bodyPart];
            if (bodyPartIndex > 0)
            {
                poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
                poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
                poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
            }
            else
            {
                poseKeypoints[baseOffset] = 0.f;
                poseKeypoints[baseOffset + 1] = 0.f;
                poseKeypoints[baseOffset + 2] = 0.f;
            }
        }
    }
}

void nms_kernel(float* ptr, float* top_ptr, int h, int w, int max_peaks, float threshold){
    int num_peaks = 0;
    for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y){
        for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x){
            float value = ptr[y*w + x];
            if (1 < x && x < (w-2) && 1 < y && y < (h-2)){
                if (value > threshold){
                    const float topLeft = ptr[(y - 1)*w + x - 1];
                    const float top = ptr[(y - 1)*w + x];
                    const float topRight = ptr[(y - 1)*w + x + 1];
                    const float left = ptr[y*w + x - 1];
                    const float right = ptr[y*w + x + 1];
                    const float bottomLeft = ptr[(y + 1)*w + x - 1];
                    const float bottom = ptr[(y + 1)*w + x];
                    const float bottomRight = ptr[(y + 1)*w + x + 1];

                    if (value > topLeft && value > top && value > topRight
                        && value > left && value > right
                        && value > bottomLeft && value > bottom && value > bottomRight)
                    {
                        float xAcc = 0;
                        float yAcc = 0;
                        float scoreAcc = 0;
                        for (int kx = -3; kx <= 3; ++kx){
                            int ux = x + kx;
                            if (ux >= 0 && ux < w){
                                for (int ky = -3; ky <= 3; ++ky){
                                    int uy = y + ky;
                                    if (uy >= 0 && uy < h){
                                        float score = ptr[uy * w + ux];
                                        xAcc += ux * score;
                                        yAcc += uy * score;
                                        scoreAcc += score;
                                    }
                                }
                            }
                        }

                        xAcc /= scoreAcc;
                        yAcc /= scoreAcc;
                        scoreAcc = value;
                        top_ptr[(num_peaks + 1) * 3 + 0] = xAcc + 0.5;
                        top_ptr[(num_peaks + 1) * 3 + 1] = yAcc + 0.5;
                        top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
                        num_peaks++;
                    }
                }
            }
            else if (x == 1 || x == (w-2) || y == 1 || y == (h-2)){
                if (value > threshold){
                    const auto topLeft      = ((0 < x && 0 < y)         ? ptr[(y-1)*w + x-1]  : threshold);
                    const auto top          = (0 < y                    ? ptr[(y-1)*w + x]    : threshold);
                    const auto topRight     = ((0 < y && x < (w-1))     ? ptr[(y-1)*w + x+1]  : threshold);
                    const auto left         = (0 < x                    ? ptr[    y*w + x-1]  : threshold);
                    const auto right        = (x < (w-1)                ? ptr[y*w + x+1]      : threshold);
                    const auto bottomLeft   = ((y < (h-1) && 0 < x)     ? ptr[(y+1)*w + x-1]  : threshold);
                    const auto bottom       = (y < (h-1)                ? ptr[(y+1)*w + x]    : threshold);
                    const auto bottomRight  = ((x < (w-1) && y < (h-1)) ? ptr[(y+1)*w + x+1]  : threshold);
                    if (value > topLeft && value > top && value > topRight
                        && value > left && value > right
                        && value > bottomLeft && value > bottom && value > bottomRight)
                    {
                        float xAcc = 0;
                        float yAcc = 0;
                        float scoreAcc = 0;
                        for (int kx = -3; kx <= 3; ++kx){
                            int ux = x + kx;
                            if (ux >= 0 && ux < w){
                                for (int ky = -3; ky <= 3; ++ky){
                                    int uy = y + ky;
                                    if (uy >= 0 && uy < h){
                                        float score = ptr[uy * w + ux];
                                        xAcc += ux * score;
                                        yAcc += uy * score;
                                        scoreAcc += score;
                                    }
                                }
                            }
                        }

                        xAcc /= scoreAcc;
                        yAcc /= scoreAcc;
                        scoreAcc = value;
                        top_ptr[(num_peaks + 1) * 3 + 0] = xAcc + 0.5;
                        top_ptr[(num_peaks + 1) * 3 + 1] = yAcc + 0.5;
                        top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
                        num_peaks++;
                    }
                }
            }

        }
    }
    top_ptr[0] = num_peaks;
}

void OpenPose::nms(BlobData* bottom_blob, BlobData* top_blob, float threshold){

    int w = bottom_blob->width;
    int h = bottom_blob->height;
    int plane_offset = w * h;
    float* ptr = bottom_blob->list;
    float* top_ptr = top_blob->list;
    int top_plane_offset = top_blob->width * top_blob->height;
    int max_peaks = top_blob->height - 1;
    memset(top_ptr, 0, top_blob->count * sizeof(float));

#pragma omp parallel for num_threads(8)
    for (int c = 0; c < top_blob->channels - 1; ++c){
        nms_kernel(ptr + c * plane_offset, top_ptr + c * top_plane_offset, h, w, max_peaks, threshold);
    }
}


void OpenPose::renderKeypointsCpu(Mat& frame, vector<float>& keypoints, vector<int> keyshape, std::vector<unsigned int>& pairs,
                                  std::vector<float> colors, float thicknessCircleRatio, float thicknessLineRatioWRTCircle,
                                  float threshold, float scale)
{
    // Get frame channels
    const auto width = frame.cols;
    const auto height = frame.rows;
    const auto area = width * height;
    
    float alpha = 0.8;
    Mat mask(frame.rows, frame.cols, CV_8UC3, Scalar(0,0,0));
    
    // Parameters
    const auto lineType = 8;
    const auto shift = 0;
    const auto numberColors = colors.size();
    const auto thresholdRectangle = 0.1f;
    const auto numberKeypoints = keyshape[1];

    // Keypoints
    for (auto person = 0; person < keyshape[0]; person++)
    {
        {
            const auto ratioAreas = 1;
            // Size-dependent variables
            const auto thicknessRatio = fastMax(intRound(std::sqrt(area)*thicknessCircleRatio * ratioAreas), 1);
            // Negative thickness in cv::circle means that a filled circle is to be drawn.
            const auto thicknessCircle = (ratioAreas > 0.05 ? thicknessRatio : -1);
            const auto thicknessLine = intRound(thicknessRatio * thicknessLineRatioWRTCircle);
            const auto radius = thicknessRatio / 4;

            // Draw lines
            for (auto pair = 0u; pair < pairs.size(); pair += 2)
            {
                const auto index1 = (person * numberKeypoints + pairs[pair]) * keyshape[2];
                const auto index2 = (person * numberKeypoints + pairs[pair + 1]) * keyshape[2];
                if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
                {
                    const auto colorIndex = pairs[pair + 1] * 3; // Before: colorIndex = pair/2*3;
                    const cv::Scalar color{ colors[(colorIndex+2) % numberColors],
                                            colors[(colorIndex + 1) % numberColors],
                                            colors[(colorIndex + 0) % numberColors]};
                    const cv::Point keypoint1{ intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale) };
                    const cv::Point keypoint2{ intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale) };
                    cv::line(mask, keypoint1, keypoint2, color, thicknessLine, lineType, shift);
                }
            }

            // Draw circles
            for (auto part = 0; part < numberKeypoints; part++)
            {
                const auto faceIndex = (person * numberKeypoints + part) * keyshape[2];
                if (keypoints[faceIndex + 2] > threshold)
                {
                    const auto colorIndex = part * 3;
                    const cv::Scalar color{ colors[(colorIndex + 2) % numberColors],
                                            colors[(colorIndex + 1) % numberColors],
                                            colors[(colorIndex + 0) % numberColors]};
                    const cv::Point center{ intRound(keypoints[faceIndex] * scale), intRound(keypoints[faceIndex + 1] * scale) };
                    cv::circle(mask, center, radius, color, thicknessCircle, lineType, shift);
                }
            }
        }
    }
    addWeighted(mask, alpha, frame, 1, 1, frame);
}

void OpenPose::renderPoseKeypointsCpu(Mat& frame, vector<float>& poseKeypoints, vector<int> keyshape,
                                      float renderThreshold, float scale, bool blendOriginalFrame)
{
    // Background
    if (!blendOriginalFrame)
        frame.setTo(0.f); // [0-255]

    // Parameters
    auto thicknessCircleRatio = 1.f / 100.f;
    auto thicknessLineRatioWRTCircle = 0.6f;
    auto& pairs = POSE_COCO_PAIRS_RENDER;

    // Render keypoints
    renderKeypointsCpu(frame, poseKeypoints, keyshape, pairs, POSE_COCO_COLORS_RENDER, thicknessCircleRatio,
                       thicknessLineRatioWRTCircle, renderThreshold, scale);
}

BlobData* OpenPose::createBlob_local(int num, int channels, int height, int width){
    BlobData* blob = new BlobData();
    blob->num = num;
    blob->width = width;
    blob->channels = channels;
    blob->height = height;
    blob->count = num*width*channels*height;
    blob->list = new float[blob->count];
    return blob;
}

void OpenPose::releaseBlob_local(BlobData** blob){
    if (blob){
        BlobData* ptr = *blob;
        if (ptr){
            if (ptr->list)
                delete[] ptr->list;

            delete ptr;
        }
        *blob = 0;
    }
}

void OpenPose::init(vector<int>& _modelInputShape,
                    int _resizeScalar,
                    float _nmsThreshold,
                    float _interMinAboveThreshold,
                    float _interThreshold,
                    int _minSubsetCnt,
                    float _minSubsetScore
)
{
    interMinAboveThreshold = _interMinAboveThreshold;
    interThreshold = _interThreshold;
    minSubsetCnt = _minSubsetCnt;
    minSubsetScore = _minSubsetScore;
    modelInputShape = _modelInputShape;
    resizeScalar = _resizeScalar;
    nmsThreshold = _nmsThreshold;
    peaks = createBlob_local(modelInputShape[0], 25, POSE_MAX_PEOPLE, 3);
    net_output = createBlob_local(modelInputShape[0], 78, modelInputShape[2] / 8, modelInputShape[3] / 8);
    heapmaps = createBlob_local(modelInputShape[0], 78, net_output->height * resizeScalar, net_output->width * resizeScalar);
    baseSize = Size(net_output->width * resizeScalar, net_output->height * resizeScalar);
}

void OpenPose::deInit() {
    releaseBlob_local(&heapmaps);
    releaseBlob_local(&peaks);
}

Mat OpenPose::preprocess(Mat& img){
    return resizeFixedAspectRatio(img, Size(modelInputShape[3],modelInputShape[2]));
}

void OpenPose::postprocess(Mat& img, float* netOutputPtr) {
    net_output->list = netOutputPtr;

    vector<float> keypoints;
    vector<float> poseScores;
    vector<int> shape;
#ifdef PRINT_TIME
    struct timeval begin;
    struct timeval end;
    gettimeofday(&begin,nullptr);
#endif

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < net_output->channels; i++){
        Mat um(baseSize.height, baseSize.width, CV_32FC1, heapmaps->list + baseSize.height * baseSize.width * i);
        resize(Mat(net_output->height, net_output->width, CV_32FC1, net_output->list + net_output->width * net_output->height * i), um, baseSize, 0, 0, INTER_CUBIC);
    }
#ifdef PRINT_TIME
    gettimeofday(&end,nullptr);
    cout<<"resize time : "<<(end.tv_sec-begin.tv_sec)*1000+(end.tv_usec-begin.tv_usec) / 1000.0 <<"ms"<<endl;
    gettimeofday(&begin,nullptr);
#endif
    nms(heapmaps, peaks, nmsThreshold);
#ifdef PRINT_TIME
    gettimeofday(&end,nullptr);
    cout<<"nms time : "<<(end.tv_sec-begin.tv_sec)*1000+(end.tv_usec-begin.tv_usec) / 1000.0 <<"ms"<<endl;
    gettimeofday(&begin,nullptr);
#endif
    connectBodyPartsCpu(keypoints, 
                        heapmaps->list, 
                        peaks->list, baseSize, 
                        POSE_MAX_PEOPLE - 1, 
                        interMinAboveThreshold, 
                        interThreshold, 
                        minSubsetCnt, 
                        minSubsetScore, 
                        scale, 
                        shape);
#ifdef PRINT_TIME
    gettimeofday(&end,nullptr);
    cout<<"connectBodyPartsCpu time : "<<(end.tv_sec-begin.tv_sec)*1000+(end.tv_usec-begin.tv_usec) / 1000.0 <<"ms"<<endl;
#endif

    //mul scale
    for(int i=0; i < keypoints.size(); i+=3)
    {
        keypoints[i] = keypoints[i] * (8 / resizeScalar);
        keypoints[i+1] = keypoints[i+1] * (8 / resizeScalar);
    }

    if(keypoints.size() != 0)
    {
#ifdef PRINT_TIME
        gettimeofday(&begin,nullptr);
#endif
        renderPoseKeypointsCpu(img, keypoints, shape, 0.05, 1);
#ifdef PRINT_TIME
        gettimeofday(&end,nullptr);
        cout<<"renderPoseKeypointsCpu time : "<<(end.tv_sec-begin.tv_sec)*1000+(end.tv_usec-begin.tv_usec) / 1000.0 <<"ms"<<endl
#endif
        cout << "detected people:" << shape[0] << endl;;
        // imshow("openpose", img);
    }
}
