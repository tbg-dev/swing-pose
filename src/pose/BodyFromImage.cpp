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

        // Configuring OpenPose
        LOG_TRACE(LGR, "Configuring OpenPose...");
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        //if (FLAGS_disable_multi_thread) {
        //    opWrapper.disableMultiThreading();
        //}

        // Starting OpenPose
        LOG_TRACE(LGR, "Starting thread(s)...");
        opWrapper.start();

        // Process and display image
        LOG_DEBUG(LGR, "Processing image: " << filename);
        const cv::Mat cvImageToProcess = cv::imread(filename);
        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
        if (datumProcessed != nullptr) {
            printKeypoints(datumProcessed);
            //if (!FLAGS_no_display) {
            //    display(datumProcessed);
            //}
        } else {
            LOG_WARN(LGR, "Image could not be processed.");
        }

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

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
