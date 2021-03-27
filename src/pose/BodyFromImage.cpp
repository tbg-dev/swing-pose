#include "BodyFromImage.h"

#include <c4a/core/log.h>
#include <opencv2/opencv.hpp>

#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

static constexpr auto LGR = "pose::BodyFromImage";

namespace c4a {
namespace pose {

//-----------------------------------------------------------------------------
template<typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

//-----------------------------------------------------------------------------
template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

//-----------------------------------------------------------------------------
template<class T>
inline T fastTruncate(T value, T min = 0, T max = 1)
{
    return fastMin(max, fastMax(min, value));
}

//-----------------------------------------------------------------------------
template<typename T>
inline int positiveIntRound(const T a)
{
    return int(a+0.5f);
}

//-----------------------------------------------------------------------------
static void resizeFixedAspectRatio(
    cv::Mat& resizedCvMat, const cv::Mat& cvMat, const double scaleFactor, const cv::Point& targetSize,
    const int borderMode = cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar{0,0,0})
{
    try {
        const cv::Size cvTargetSize{targetSize.x, targetSize.y};
        cv::Mat M = cv::Mat::eye(2,3,CV_64F);
        M.at<double>(0,0) = scaleFactor;
        M.at<double>(1,1) = scaleFactor;
        if (scaleFactor != 1. || cvTargetSize != cvMat.size())
            cv::warpAffine(cvMat, resizedCvMat, M, cvTargetSize,
                           (scaleFactor > 1. ? cv::INTER_CUBIC : cv::INTER_AREA), borderMode, borderValue);
        else
            cvMat.copyTo(resizedCvMat);
    } catch (const std::exception& e) {
        LOG_ERROR(LGR, e.what());
    }
}

//-----------------------------------------------------------------------------
IBodyFromImagePtr IBodyFromImage::create()
{
    return std::make_shared<BodyFromImage>();
}

//-----------------------------------------------------------------------------
int BodyFromImage::run(const char* filename)
{
    try {
        LOG_TRACE(LGR, "Starting OpenPose demo...");
        const auto opTimer = op::getTimerInit();

        // Read the image
        const cv::Mat cvInputData = cv::imread(filename);

        // Sanity checks
        if (cvInputData.empty())
            LOG_ERROR(LGR, "Input image is empty.");
        if (cvInputData.channels() != 3)
            LOG_ERROR(LGR, "Input images must be 3-channel BGR.");

        // Set poseNetInputSize
        auto poseNetInputSize = cv::Size{-1, 368};
        if (poseNetInputSize.width <= 0 || poseNetInputSize.height <= 0) {
            if (poseNetInputSize.width <= 0)
                poseNetInputSize.width = 16 * positiveIntRound(1 / 16.f * poseNetInputSize.height * cvInputData.cols / (float)cvInputData.rows);
            else // if (poseNetInputSize.height <= 0)
                poseNetInputSize.height = 16 * positiveIntRound(1 / 16.f * poseNetInputSize.width * cvInputData.rows / (float)cvInputData.cols);
        }

        // scaleInputToNetInputs & netInputSizes - Rescale keeping aspect ratio
        const int scalesNumber = 1;
        const float scaleGap = 0.25f;
        std::vector<double> scaleInputToNetInputs(scalesNumber, 1.f);
        std::vector<cv::Point> netInputSizes(scalesNumber);
        for (auto i = 0; i < scalesNumber; i++) {
            const auto currentScale = 1. - i*scaleGap;
            if (currentScale < 0. || 1. < currentScale)
                LOG_ERROR(LGR,"All scales must be in the range [0, 1], i.e., 0 <= 1-scale_number*scale_gap <= 1");

            const auto targetWidth = fastTruncate(
                    positiveIntRound(poseNetInputSize.width * currentScale) / 16 * 16, 1, poseNetInputSize.width);
            const auto targetHeight = fastTruncate(
                    positiveIntRound(poseNetInputSize.height * currentScale) / 16 * 16, 1, poseNetInputSize.height);
            const auto ratioWidth = (targetWidth - 1) / (double)(cvInputData.cols - 1);
            const auto ratioHeight = (targetHeight - 1) / (double)(cvInputData.rows - 1);
            scaleInputToNetInputs[i] = fastMin(ratioWidth, ratioHeight);
            netInputSizes[i] = cv::Point{targetWidth, targetHeight};
        }

        // scaleInputToOutput - Scale between input and desired output size
        cv::Point netOutputSize{cvInputData.cols, cvInputData.rows};
        double scaleInputToOutput = 1.;

        // inputNetData - Rescale keeping aspect ratio and transform to float the input deep net image
        std::vector<op::Array<float>> inputNetData(scalesNumber);
        for (auto i = 0u ; i < inputNetData.size() ; i++) {
            cv::Mat frameWithNetSize;
            resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);
            // Fill inputNetData[i]
            inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
            op::uCharCvMatToFloatPtr(inputNetData[i].getPtr(), OP_CV2OPMAT(frameWithNetSize), 1);

            // // OpenCV equivalent
            // const auto scale = 1/255.;
            // const cv::Scalar mean{128,128,128};
            // const cv::Size outputSize{netInputSizes[i].x, netInputSizes[i].y};
            // // cv::Mat cvMat;
            // cv::dnn::blobFromImage(
            //     // frameWithNetSize, cvMat, scale, outputSize, mean);
            //     frameWithNetSize, inputNetData[i].getCvMat(), scale, outputSize, mean);
            // // opLog(cv::norm(cvMat - inputNetData[i].getCvMat())); // ~0.25
        }

        // outputData - Rescale keeping aspect ratio and transform to float the output image
        op::Array<float> outputData({netOutputSize.y, netOutputSize.x, 3}); // This size is used everywhere
        cv::Mat frameWithOutputSize;
        resizeFixedAspectRatio(frameWithOutputSize, cvInputData, scaleInputToOutput, netOutputSize);
        frameWithOutputSize.convertTo(OP_OP2CVMAT(outputData.getCvMat()), CV_32FC3);

        op::Array<float> poseNetOutput;
        op::Array<float> poseKeypoints;
        {
            const op::Point<int> inputSize{cvInputData.cols, cvInputData.rows};
            const op::WrapperStructPose wrapperStructPose;
            const auto modelFolder = op::formatAsDirectory(wrapperStructPose.modelFolder.getStdString());
            const auto poseExtractorNet = std::make_shared<op::PoseExtractorCaffe>(
                    wrapperStructPose.poseModel, modelFolder, wrapperStructPose.gpuNumberStart,
                    wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScaleMode,
                    wrapperStructPose.addPartCandidates, wrapperStructPose.maximizePositives,
                    wrapperStructPose.protoTxtPath.getStdString(),
                    wrapperStructPose.caffeModelPath.getStdString(),
                    wrapperStructPose.upsamplingRatio, wrapperStructPose.poseMode == op::PoseMode::Enabled,
                    wrapperStructPose.enableGoogleLogging);
            const auto poseExtractor = std::make_shared<op::PoseExtractor>(poseExtractorNet);
            poseExtractor->initializationOnThread();
            poseExtractor->forwardPass(inputNetData, inputSize, scaleInputToNetInputs, poseNetOutput);
            poseKeypoints = poseExtractor->getPoseKeypoints().clone();
        }

        // Rescale keypoints to output size
        auto poseKeypointsRescaled = poseKeypoints.clone();
        if (!poseKeypointsRescaled.empty() && (float)scaleInputToOutput != float(1)) {
            // Error check
            if (poseKeypointsRescaled.getSize(2) != 3 && poseKeypointsRescaled.getSize(2) != 4)
                LOG_ERROR(LGR, "The Array<T> is not a (x,y,score) or (x,y,z,score) format array. This"
                               " function is only for those 2 dimensions: [sizeA x sizeB x 3or4].");
            // Get #people and #parts
            const auto numberPeople = poseKeypointsRescaled.getSize(0);
            const auto numberParts = poseKeypointsRescaled.getSize(1);
            const auto xyzChannels = poseKeypointsRescaled.getSize(2);
            // For each person
            for (auto person = 0 ; person < numberPeople ; person++) {
                // For each body part
                for (auto part = 0 ; part < numberParts ; part++) {
                    const auto finalIndex = xyzChannels*(person*numberParts + part);
                    for (auto xyz = 0 ; xyz < xyzChannels-1 ; xyz++)
                        poseKeypointsRescaled[finalIndex+xyz] *= scaleInputToOutput;
                }
            }
        }

        // Render keypoints
        if (!outputData.empty()) {
            // Background
            const bool blendOriginalFrame = true;
            if (!blendOriginalFrame)
                outputData.getCvMat().setTo(0.f); // [0-255]

            // Parameters
            const auto thicknessCircleRatio = 1.f / 75.f;
            const auto thicknessLineRatioWRTCircle = 0.75f;
            const auto &pairs = op::getPoseBodyPartPairsRender(op::PoseModel::BODY_25);
            const auto &poseScales = op::getPoseScales(op::PoseModel::BODY_25);
            const auto &colors = op::getPoseColors(op::PoseModel::BODY_25);

            // Array<T> --> cv::Mat
            auto frame = outputData.getCvMat();
            cv::Mat cvFrame = OP_OP2CVMAT(frame);

            // Sanity check
            const std::string errorMessage = "The Array<T> is not a RGB image or 3-channel keypoint array. This function"
                                             " is only for array of dimension: [sizeA x sizeB x 3].";
            if (cvFrame.channels() != 3)
                LOG_ERROR(LGR, errorMessage);

            // Get frame channels
            const auto width = cvFrame.size[1];
            const auto height = cvFrame.size[0];
            const auto area = width * height;
            cv::Mat frameBGR(height, width, CV_32FC3, cvFrame.data);

            // Parameters
            const auto lineType = 8;
            const auto shift = 0;
            const auto numberColors = colors.size();
            const auto numberScales = poseScales.size();
            const auto thresholdRectangle = float(0.1);
            const auto numberKeypoints = poseKeypoints.getSize(1);
            const float renderThreshold = 0.05f;

            // Keypoints
            for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++) {
                const auto personRectangle = getKeypointsRectangle(poseKeypoints, person, thresholdRectangle);
                if (personRectangle.area() > 0) {
                    const auto ratioAreas = fastMin(
                            float(1), fastMax(
                                    personRectangle.width/(float)width, personRectangle.height/(float)height));
                    // Size-dependent variables
                    const auto thicknessRatio = fastMax(
                            positiveIntRound(std::sqrt(area)* thicknessCircleRatio * ratioAreas), 2);
                    // Negative thickness in cv::circle means that a filled circle is to be drawn.
                    const auto thicknessCircle = fastMax(1, (ratioAreas > float(0.05) ? thicknessRatio : -1));
                    const auto thicknessLine = fastMax(
                            1, positiveIntRound(thicknessRatio * thicknessLineRatioWRTCircle));
                    const auto radius = thicknessRatio / 2;

                    // Draw lines
                    for (auto pair = 0u ; pair < pairs.size() ; pair+=2) {
                        const auto index1 = (person * numberKeypoints + pairs[pair]) * poseKeypoints.getSize(2);
                        const auto index2 = (person * numberKeypoints + pairs[pair+1]) * poseKeypoints.getSize(2);
                        if (poseKeypoints[index1+2] > renderThreshold && poseKeypoints[index2+2] > renderThreshold) {
                            const auto thicknessLineScaled = positiveIntRound(
                                    thicknessLine * poseScales[pairs[pair+1] % numberScales]);
                            const auto colorIndex = pairs[pair+1]*3; // Before: colorIndex = pair/2*3;
                            const cv::Scalar color{
                                    colors[(colorIndex+2) % numberColors],
                                    colors[(colorIndex+1) % numberColors],
                                    colors[colorIndex % numberColors]
                            };
                            const cv::Point keypoint1{
                                    positiveIntRound(poseKeypoints[index1]), positiveIntRound(poseKeypoints[index1+1])};
                            const cv::Point keypoint2{
                                    positiveIntRound(poseKeypoints[index2]), positiveIntRound(poseKeypoints[index2+1])};
                            cv::line(frameBGR, keypoint1, keypoint2, color, thicknessLineScaled, lineType, shift);
                        }
                    }

                    // Draw circles
                    for (auto part = 0 ; part < numberKeypoints ; part++) {
                        const auto faceIndex = (person * numberKeypoints + part) * poseKeypoints.getSize(2);
                        if (poseKeypoints[faceIndex+2] > renderThreshold) {
                            const auto radiusScaled = positiveIntRound(radius * poseScales[part % numberScales]);
                            const auto thicknessCircleScaled = positiveIntRound(
                                    thicknessCircle * poseScales[part % numberScales]);
                            const auto colorIndex = part*3;
                            const cv::Scalar color{
                                    colors[(colorIndex+2) % numberColors],
                                    colors[(colorIndex+1) % numberColors],
                                    colors[colorIndex % numberColors]
                            };
                            const cv::Point center{positiveIntRound(poseKeypoints[faceIndex]),
                                                   positiveIntRound(poseKeypoints[faceIndex+1])};
                            cv::circle(frameBGR, center, radiusScaled, color, thicknessCircleScaled, lineType,
                                       shift);
                        }
                    }
                }
            }
        }

        cv::Mat cvOutputData;
        const cv::Mat constCvMat = OP_OP2CVCONSTMAT(outputData.getConstCvMat());
        constCvMat.convertTo(cvOutputData, CV_8UC3);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Display the result
        if (!cvOutputData.empty()) {
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvOutputData);
            cv::waitKey(0);
        }

        // Return success
        return 0;

    } catch (const std::exception& e) {
        // Return failure
        LOG_ERROR(LGR, "Exception: " << e.what());
        return -1;
    }
}

} // namespace pose
} // namespace c4a
