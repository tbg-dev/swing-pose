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
        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);

        // Process the image
        LOG_DEBUG(LGR, "Processing image: " << filename);
        using TDatum = op::BASE_DATUM;
        auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<TDatum>>>();
        datumsPtr->emplace_back();
        auto& tDatumPtr = datumsPtr->at(0);
        tDatumPtr = std::make_shared<TDatum>();
        tDatumPtr->cvInputData = imageToProcess;
        const op::Point<int> inputSize{tDatumPtr->cvInputData.cols(), tDatumPtr->cvInputData.rows()};
        std::tie(tDatumPtr->scaleInputToNetInputs, tDatumPtr->netInputSizes, tDatumPtr->scaleInputToOutput,
                 tDatumPtr->netOutputSize) = scaleAndSizeExtractor->extract(inputSize);
        tDatumPtr->inputNetData = cvMatToOpInput->createArray(
                tDatumPtr->cvInputData, tDatumPtr->scaleInputToNetInputs, tDatumPtr->netInputSizes);
        tDatumPtr->outputData = cvMatToOpOutput->createArray(
                tDatumPtr->cvInputData, tDatumPtr->scaleInputToOutput, tDatumPtr->netOutputSize);
        poseExtractor->forwardPass(
                tDatumPtr->inputNetData, op::Point<int>{tDatumPtr->cvInputData.cols(), tDatumPtr->cvInputData.rows()},
                tDatumPtr->scaleInputToNetInputs, tDatumPtr->poseNetOutput, tDatumPtr->id);
        tDatumPtr->poseCandidates = poseExtractor->getCandidatesCopy();
        tDatumPtr->poseHeatMaps = poseExtractor->getHeatMapsCopy();
        tDatumPtr->poseKeypoints = poseExtractor->getPoseKeypoints().clone();
        tDatumPtr->poseScores = poseExtractor->getPoseScores().clone();
        tDatumPtr->scaleNetToOutput = poseExtractor->getScaleNetToOutput();
        tDatumPtr->elementRendered = poseRenderer->renderPose(
                tDatumPtr->outputData, tDatumPtr->poseKeypoints, (float)tDatumPtr->scaleInputToOutput,
                (float)tDatumPtr->scaleNetToOutput);
        tDatumPtr->cvOutputData = opOutputToCvMat->formatToCvMat(tDatumPtr->outputData);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Display the result
        printKeypoints(datumsPtr);
        display(datumsPtr);

        // Return success
        return 0;

    } catch (const std::exception& e) {
        // Return failure
        LOG_ERROR(LGR, "Exception: " << e.what());
        return -1;
    }
}

//-----------------------------------------------------------------------------
void BodyFromImage::display(const std::shared_ptr<std::vector<op::DatumPtr>>& datumsPtr)
{
    try {
        // User's displaying/saving/other processing here
        // datum.cvOutputData: rendered frame with pose or heatmaps
        // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty()) {
            // Display image
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty()) {
                cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
                cv::waitKey(0);
            } else {
                LOG_WARN(LGR, "Empty cv::Mat as output.");
            }
        } else {
            LOG_WARN(LGR, "Nullptr or empty datumsPtr found.");
        }

    } catch (const std::exception& e) {
        LOG_ERROR(LGR, "Exception: " << e.what());
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

//-----------------------------------------------------------------------------
void BodyFromImage::printKeypoints(const std::shared_ptr<std::vector<op::DatumPtr>>& datumsPtr)
{
    try {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty()) {
            // Alternative 1
            LOG_INFO(LGR, "Body keypoints: " << datumsPtr->at(0)->poseKeypoints.toString());

            // // Alternative 2
            // op::opLog(datumsPtr->at(0)->poseKeypoints, op::Priority::High);

            // // Alternative 3
            // std::cout << datumsPtr->at(0)->poseKeypoints << std::endl;

            // // Alternative 4 - Accesing each element of the keypoints
            // op::opLog("\nKeypoints:", op::Priority::High);
            // const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            // op::opLog("Person pose keypoints:", op::Priority::High);
            // for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
            // {
            //     op::opLog("Person " + std::to_string(person) + " (x, y, score):", op::Priority::High);
            //     for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
            //     {
            //         std::string valueToPrint;
            //         for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
            //             valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
            //         op::opLog(valueToPrint, op::Priority::High);
            //     }
            // }
            // op::opLog(" ", op::Priority::High);
        } else {
            LOG_WARN(LGR, "Nullptr or empty datumsPtr found.");
        }

    } catch (const std::exception& e) {
        LOG_ERROR(LGR, "Exception: " << e.what());
    }
}

} // namespace pose
} // namespace c4a
