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
static void resizeFixedAspectRatio(
    cv::Mat& resizedCvMat, const cv::Mat& cvMat, const double scaleFactor, const op::Point<int>& targetSize,
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

        // Create the OpenPose workers
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
        const auto poseRenderer = std::make_shared<op::PoseCpuRenderer>(
            wrapperStructPose.poseModel, wrapperStructPose.renderThreshold,
            wrapperStructPose.blendOriginalFrame, wrapperStructPose.alphaKeypoint,
            wrapperStructPose.alphaHeatMap, wrapperStructPose.defaultPartToRender);

        poseExtractor->initializationOnThread();
        poseRenderer->initializationOnThread();

        // Sanity checks
        if (wrapperStructPose.netInputSize.x <= 0 && wrapperStructPose.netInputSize.y <= 0)
            LOG_ERROR(LGR,"Only 1 of the dimensions of net input resolution can be <= 0.");
        if ((wrapperStructPose.netInputSize.x > 0 && wrapperStructPose.netInputSize.x % 16 != 0)
            || (wrapperStructPose.netInputSize.y > 0 && wrapperStructPose.netInputSize.y % 16 != 0))
            LOG_ERROR(LGR, "Net input resolution must be multiples of 16.");
        if (wrapperStructPose.scalesNumber < 1)
            LOG_ERROR(LGR, "There must be at least 1 scale.");
        if (wrapperStructPose.scaleGap <= 0.)
            LOG_ERROR(LGR, "The gap between scales must be strictly positive.");

        // Read the image
        const cv::Mat cvInputData = cv::imread(filename);
        const op::Point<int> inputSize{cvInputData.cols, cvInputData.rows};

        // Sanity checks
        if (cvInputData.empty())
            LOG_ERROR(LGR, "Input image is empty.");
        if (cvInputData.channels() != 3)
            LOG_ERROR(LGR, "Input images must be 3-channel BGR.");

        // Set poseNetInputSize
        auto poseNetInputSize = wrapperStructPose.netInputSize;
        if (poseNetInputSize.x <= 0 || poseNetInputSize.y <= 0) {
            if (poseNetInputSize.x <= 0)
                poseNetInputSize.x = 16 * op::positiveIntRound(1 / 16.f * poseNetInputSize.y * inputSize.x / (float)inputSize.y);
            else // if (poseNetInputSize.y <= 0)
                poseNetInputSize.y = 16 * op::positiveIntRound(1 / 16.f * poseNetInputSize.x * inputSize.y / (float)inputSize.x);
        }

        // scaleInputToNetInputs & netInputSizes - Rescale keeping aspect ratio
        std::vector<double> scaleInputToNetInputs(wrapperStructPose.scalesNumber, 1.f);
        std::vector<op::Point<int>> netInputSizes(wrapperStructPose.scalesNumber);
        for (auto i = 0; i < wrapperStructPose.scalesNumber; i++) {
            const auto currentScale = 1. - i*wrapperStructPose.scaleGap;
            if (currentScale < 0. || 1. < currentScale)
                LOG_ERROR(LGR,"All scales must be in the range [0, 1], i.e., 0 <= 1-scale_number*scale_gap <= 1");

            const auto targetWidth = op::fastTruncate(
                    op::positiveIntRound(poseNetInputSize.x * currentScale) / 16 * 16, 1, poseNetInputSize.x);
            const auto targetHeight = op::fastTruncate(
                    op::positiveIntRound(poseNetInputSize.y * currentScale) / 16 * 16, 1, poseNetInputSize.y);
            const op::Point<int> targetSize{targetWidth, targetHeight};
            scaleInputToNetInputs[i] = op::resizeGetScaleFactor(inputSize, targetSize);
            netInputSizes[i] = targetSize;
        }

        // scaleInputToOutput - Scale between input and desired output size
        op::Point<int> netOutputSize;
        double scaleInputToOutput;
        if (wrapperStructPose.outputSize.x > 0 && wrapperStructPose.outputSize.y > 0) {
            netOutputSize = wrapperStructPose.outputSize;
            scaleInputToOutput = op::resizeGetScaleFactor(inputSize, wrapperStructPose.outputSize);
        } else {
            netOutputSize = inputSize;
            scaleInputToOutput = 1.;
        }

        // inputNetData - Rescale keeping aspect ratio and transform to float the input deep net image
        std::vector<op::Array<float>> inputNetData(wrapperStructPose.scalesNumber);
        for (auto i = 0u ; i < inputNetData.size() ; i++) {
            cv::Mat frameWithNetSize;
            resizeFixedAspectRatio(frameWithNetSize, cvInputData, scaleInputToNetInputs[i], netInputSizes[i]);
            // Fill inputNetData[i]
            inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
            uCharCvMatToFloatPtr(
                    inputNetData[i].getPtr(), OP_CV2OPMAT(frameWithNetSize),
                    (wrapperStructPose.poseModel == op::PoseModel::BODY_19N ? 2 : 1));

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
        poseExtractor->forwardPass(inputNetData, inputSize, scaleInputToNetInputs, poseNetOutput);
        const std::vector<std::vector<std::array<float,3>>> poseCandidates = poseExtractor->getCandidatesCopy();
        const op::Array<float> poseHeatMaps = poseExtractor->getHeatMapsCopy();
        const op::Array<float> poseKeypoints = poseExtractor->getPoseKeypoints().clone();
        const op::Array<float> poseScores = poseExtractor->getPoseScores().clone();
        const double scaleNetToOutput = poseExtractor->getScaleNetToOutput();
        const std::pair<int, std::string> elementRendered = poseRenderer->renderPose(
                outputData, poseKeypoints, (float)scaleInputToOutput,
                (float)scaleNetToOutput);

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
