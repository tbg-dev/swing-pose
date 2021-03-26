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
        const auto scaleAndSizeExtractor = std::make_shared<op::ScaleAndSizeExtractor>(
            wrapperStructPose.netInputSize, wrapperStructPose.outputSize,
            wrapperStructPose.scalesNumber, wrapperStructPose.scaleGap);
        const auto cvMatToOpInput = std::make_shared<op::CvMatToOpInput>(wrapperStructPose.poseModel);
        const auto cvMatToOpOutput = std::make_shared<op::CvMatToOpOutput>();
        const auto opOutputToCvMat = std::make_shared<op::OpOutputToCvMat>();
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

        // Read the image
        const cv::Mat cvImageToProcess = cv::imread(filename);

        // Process the image
        const op::Matrix cvInputData = OP_CV2OPCONSTMAT(cvImageToProcess);
        const op::Point<int> inputSize{cvInputData.cols(), cvInputData.rows()};
        std::vector<double> scaleInputToNetInputs;
        std::vector<op::Point<int>> netInputSizes;
        double scaleInputToOutput;
        op::Point<int> netOutputSize;
        std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput,
                 netOutputSize) = scaleAndSizeExtractor->extract(inputSize);
        std::vector<op::Array<float>> inputNetData = cvMatToOpInput->createArray(
                cvInputData, scaleInputToNetInputs, netInputSizes);
        op::Array<float> outputData = cvMatToOpOutput->createArray(
                cvInputData, scaleInputToOutput, netOutputSize);
        op::Array<float> poseNetOutput;
        poseExtractor->forwardPass(
                inputNetData, op::Point<int>{cvInputData.cols(), cvInputData.rows()},
                scaleInputToNetInputs, poseNetOutput);
        const std::vector<std::vector<std::array<float,3>>> poseCandidates = poseExtractor->getCandidatesCopy();
        const op::Array<float> poseHeatMaps = poseExtractor->getHeatMapsCopy();
        const op::Array<float> poseKeypoints = poseExtractor->getPoseKeypoints().clone();
        const op::Array<float> poseScores = poseExtractor->getPoseScores().clone();
        const double scaleNetToOutput = poseExtractor->getScaleNetToOutput();
        const std::pair<int, std::string> elementRendered = poseRenderer->renderPose(
                outputData, poseKeypoints, (float)scaleInputToOutput,
                (float)scaleNetToOutput);
        const op::Matrix cvOutputData = opOutputToCvMat->formatToCvMat(outputData);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Display the result
        printKeypoints(poseKeypoints);
        display(cvOutputData);

        // Return success
        return 0;

    } catch (const std::exception& e) {
        // Return failure
        LOG_ERROR(LGR, "Exception: " << e.what());
        return -1;
    }
}

//-----------------------------------------------------------------------------
void BodyFromImage::display(const op::Matrix& cvOutputData)
{
    try {
        // Display the image
        const cv::Mat cvMat = OP_OP2CVCONSTMAT(cvOutputData);
        if (!cvMat.empty()) {
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
            cv::waitKey(0);
        } else {
            LOG_WARN(LGR, "Empty cv::Mat as output.");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(LGR, "Exception: " << e.what());
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

//-----------------------------------------------------------------------------
void BodyFromImage::printKeypoints(const op::Array<float>& poseKeypoints)
{
    try {
        // Print the pose keypoints
        LOG_INFO(LGR, "Body keypoints: " << poseKeypoints.toString());
    } catch (const std::exception& e) {
        LOG_ERROR(LGR, "Exception: " << e.what());
    }
}

} // namespace pose
} // namespace c4a
