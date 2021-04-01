#include "BodyFromImage.h"

#include <c4a/core/log.h>
#include <opencv2/opencv.hpp>

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
static double resizeGetScaleFactor(const cv::Size& initialSize, const cv::Size& targetSize)
{
    try {
        const auto ratioWidth = (targetSize.width - 1) / (double)(initialSize.width - 1);
        const auto ratioHeight = (targetSize.height - 1) / (double)(initialSize.height - 1);
        return fastMin(ratioWidth, ratioHeight);
    } catch (const std::exception& e) {
        LOG_ERROR(LGR, e.what());
        return 0.;
    }
}

//-----------------------------------------------------------------------------
static void resizeFixedAspectRatio(
    cv::Mat& resizedCvMat, const cv::Mat& cvMat, const double scaleFactor, const cv::Size& targetSize,
    const int borderMode = cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar{0,0,0})
{
    try {
        cv::Mat M = cv::Mat::eye(2,3,CV_64F);
        M.at<double>(0,0) = scaleFactor;
        M.at<double>(1,1) = scaleFactor;
        if (scaleFactor != 1. || targetSize != cvMat.size())
            cv::warpAffine(cvMat, resizedCvMat, M, targetSize,
                           (scaleFactor > 1. ? cv::INTER_CUBIC : cv::INTER_AREA), borderMode, borderValue);
        else
            cvMat.copyTo(resizedCvMat);
    } catch (const std::exception& e) {
        LOG_ERROR(LGR, e.what());
    }
}

//-----------------------------------------------------------------------------
std::vector<op::ArrayCpuGpu<float>*> arraySharedToPtr(
    const std::vector<std::shared_ptr<op::ArrayCpuGpu<float>>>& caffeNetOutputBlob)
{
    try
    {
        // Prepare spCaffeNetOutputBlobss
        std::vector<op::ArrayCpuGpu<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
        for (auto i = 0u ; i < caffeNetOutputBlobs.size() ; i++)
            caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
        return caffeNetOutputBlobs;
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return {};
    }
}

//-----------------------------------------------------------------------------
static op::Array<float> extractPoseKeypoints(
    const std::vector<op::Array<float>>& inputNetData, const cv::Size& inputDataSize,
    const std::vector<double>& scaleInputToNetInputs, const op::Array<float>& poseNetOutput)
{
    try {
        const op::PoseModel poseModel = op::PoseModel::BODY_25;
        const int gpuId = 0;
        const std::vector<op::HeatMapType> heatMapTypes{};
        const bool maximizePositives = false;
        const bool enableGoogleLogging = true;

        cv::Size mNetOutputSize{0, 0};
        op::Array<float> mPoseKeypoints;
        op::Array<float> mPoseScores;

        // General parameters
        std::vector<std::shared_ptr<op::Net>> spNets;
        std::shared_ptr<op::ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe{
                std::make_shared<op::ResizeAndMergeCaffe<float>>()};
        std::shared_ptr<op::NmsCaffe<float>> spNmsCaffe{std::make_shared<op::NmsCaffe<float>>()};
        std::shared_ptr<op::BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe{
                std::make_shared<op::BodyPartConnectorCaffe<float>>()};
        std::vector<std::vector<int>> mNetInput4DSizes;
        // Init with thread
        std::vector<std::shared_ptr<op::ArrayCpuGpu<float>>> spCaffeNetOutputBlobs;
        std::shared_ptr<op::ArrayCpuGpu<float>> spHeatMapsBlob;
        std::shared_ptr<op::ArrayCpuGpu<float>> spPeaksBlob;

        // Layers parameters
        const auto nmsThreshold = op::getPoseDefaultNmsThreshold(poseModel, maximizePositives);
        spNmsCaffe->setThreshold(nmsThreshold);
        spBodyPartConnectorCaffe->setPoseModel(poseModel);
        spBodyPartConnectorCaffe->setMaximizePositives(maximizePositives);
        spBodyPartConnectorCaffe->setDefaultNmsThreshold(nmsThreshold);
        spBodyPartConnectorCaffe->setInterMinAboveThreshold((int)op::getPoseDefaultConnectInterMinAboveThreshold(maximizePositives));
        spBodyPartConnectorCaffe->setInterThreshold((int)op::getPoseDefaultConnectInterThreshold(poseModel, maximizePositives));
        spBodyPartConnectorCaffe->setMinSubsetCnt((int)op::getPoseDefaultMinSubsetCnt(maximizePositives));
        spBodyPartConnectorCaffe->setMinSubsetScore((int)op::getPoseDefaultConnectMinSubsetScore(maximizePositives));

        // Deep net initialization
        {
            const std::string caffeProtoTxt{"models/pose/body_25/pose_deploy.prototxt"};
            const std::string caffeModel{"models/pose/body_25/pose_iter_584000.caffemodel"};

            // Add Caffe Net
            spNets.emplace_back(std::make_shared<op::NetCaffe>(caffeProtoTxt, caffeModel, gpuId, enableGoogleLogging));
            spNets.back()->initializationOnThread();
            spCaffeNetOutputBlobs.emplace_back((spNets.back().get())->getOutputBlobArray());

            // Resize std::vectors if required
            const auto numberScales = inputNetData.size();
            mNetInput4DSizes.resize(numberScales);

            // Process each image - Caffe deep network
            while (spNets.size() < numberScales) {
                // Add Caffe Net
                spNets.emplace_back(std::make_shared<op::NetCaffe>(caffeProtoTxt, caffeModel, gpuId, false));
                spNets.back()->initializationOnThread();
                spCaffeNetOutputBlobs.emplace_back((spNets.back().get())->getOutputBlobArray());
            }

            // Sanity check
            if (spNets.size() != spCaffeNetOutputBlobs.size())
                op::error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);

            // Initialize blobs
            spHeatMapsBlob = {std::make_shared<op::ArrayCpuGpu<float>>(1, 1, 1, 1)};
            spPeaksBlob = {std::make_shared<op::ArrayCpuGpu<float>>(1, 1, 1, 1)};
        }

        // forwardPass
        {
            // Sanity checks
            if (inputNetData.empty())
                op::error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
            for (const auto& inputNetDataI : inputNetData)
                if (inputNetDataI.empty())
                    op::error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
            if (inputNetData.size() != scaleInputToNetInputs.size())
                op::error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                          __LINE__, __FUNCTION__, __FILE__);
            if (!poseNetOutput.empty()) {
                const std::string errorMsg = ". Either use OpenPose default network (`--body 1`) or fill the"
                                             " `poseNetOutput` argument (only 1 of those 2, not both).";
                op::error("The argument poseNetOutput is not empty and you have also explicitly chosen to run"
                          " the OpenPose network" + errorMsg, __LINE__, __FUNCTION__, __FILE__);
            }

            for (auto i = 0u; i < inputNetData.size(); i++)
                spNets.at(i)->forwardPass(inputNetData[i]);

            // Reshape blobs if required
            for (auto i = 0u; i < inputNetData.size(); i++) {
                // Reshape blobs if required - For dynamic sizes (e.g., images of different aspect ratio)
                const auto changedVectors = !op::vectorsAreEqual(
                        mNetInput4DSizes.at(i), inputNetData[i].getSize());
                if (changedVectors) {
                    mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                    const auto netDescreaseFactor = op::getPoseNetDecreaseFactor(poseModel);
                    // HeatMaps extractor blob and layer
                    // Caffe modifies bottom - Heatmap gets resized
                    const auto caffeNetOutputBlobs = arraySharedToPtr(spCaffeNetOutputBlobs);
                    spResizeAndMergeCaffe->Reshape(
                            caffeNetOutputBlobs, {spHeatMapsBlob.get()},
                            netDescreaseFactor, 1.f, true, gpuId);
                    // Pose extractor blob and layer
                    spNmsCaffe->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, op::getPoseMaxPeaks(),
                                      op::getPoseNumberBodyParts(poseModel), gpuId);
                    // Pose extractor blob and layer
                    spBodyPartConnectorCaffe->Reshape({spHeatMapsBlob.get(), spPeaksBlob.get()}, gpuId);
                    // In order to resize to input size to have same results as Matlab
                    // scaleInputToNetInputs[i] vs. 1.f
                }

                // Get scale net to output (i.e., image input)
                const auto ratio = 1;
                if (changedVectors)
                    mNetOutputSize = cv::Size{
                            positiveIntRound(ratio * mNetInput4DSizes[0][3]),
                            positiveIntRound(ratio * mNetInput4DSizes[0][2])};
            }

            // Resize heat maps + merge different scales
            const auto caffeNetOutputBlobs = arraySharedToPtr(spCaffeNetOutputBlobs);

            // Set and fill floatScaleRatios
            std::vector<float> floatScaleRatios;
            std::for_each(
                    scaleInputToNetInputs.begin(), scaleInputToNetInputs.end(),
                    [&floatScaleRatios](const double value) { floatScaleRatios.emplace_back(float(value)); });
            spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
            spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {spHeatMapsBlob.get()});

            // Get scale net to output (i.e., image input)
            const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
            const cv::Size netSize{
                    positiveIntRound(scaleProducerToNetInput * inputDataSize.width),
                    positiveIntRound(scaleProducerToNetInput * inputDataSize.height)};
            const auto scaleNetToOutput = resizeGetScaleFactor(netSize, inputDataSize);
            const auto nmsOffset = float(0.5 / scaleNetToOutput);
            spNmsCaffe->setOffset(op::Point<float>{nmsOffset, nmsOffset});
            spBodyPartConnectorCaffe->setScaleNetToOutput(scaleNetToOutput);

            // Get peaks by Non-Maximum Suppression
            spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()});

            // Connecting body parts
            spBodyPartConnectorCaffe->Forward({spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);
        }

        return mPoseKeypoints.clone();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return op::Array<float>{};
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
        const cv::Size inputSize{cvInputData.cols, cvInputData.rows};
        const double inputAspectRatio = inputSize.aspectRatio();

        // Sanity checks
        if (cvInputData.empty())
            LOG_ERROR(LGR, "Input image is empty.");
        if (cvInputData.channels() != 3)
            LOG_ERROR(LGR, "Input images must be 3-channel BGR.");

        // Set poseNetInputSize
        auto poseNetInputSize = cv::Size{-1, 368};
        if (poseNetInputSize.width <= 0 || poseNetInputSize.height <= 0) {
            if (poseNetInputSize.width <= 0)
                poseNetInputSize.width = 16 * positiveIntRound(1 / 16.f * poseNetInputSize.height * inputAspectRatio);
            else // if (poseNetInputSize.height <= 0)
                poseNetInputSize.height = 16 * positiveIntRound(1 / 16.f * poseNetInputSize.width / inputAspectRatio);
        }

        // scaleInputToNetInputs & netInputSizes - Rescale keeping aspect ratio
        const int scalesNumber = 1;
        const float scaleGap = 0.25f;
        std::vector<double> scaleInputToNetInputs(scalesNumber, 1.f);
        std::vector<cv::Size> netInputSizes(scalesNumber);
        for (auto i = 0; i < scalesNumber; i++) {
            const auto currentScale = 1. - i*scaleGap;
            if (currentScale < 0. || 1. < currentScale)
                LOG_ERROR(LGR,"All scales must be in the range [0, 1], i.e., 0 <= 1-scale_number*scale_gap <= 1");

            const auto targetWidth = fastTruncate(
                    positiveIntRound(poseNetInputSize.width * currentScale) / 16 * 16, 1, poseNetInputSize.width);
            const auto targetHeight = fastTruncate(
                    positiveIntRound(poseNetInputSize.height * currentScale) / 16 * 16, 1, poseNetInputSize.height);
            const auto ratioWidth = (targetWidth - 1) / (double)(inputSize.width - 1);
            const auto ratioHeight = (targetHeight - 1) / (double)(inputSize.height - 1);
            scaleInputToNetInputs[i] = fastMin(ratioWidth, ratioHeight);
            netInputSizes[i] = cv::Size{targetWidth, targetHeight};
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
            inputNetData[i].reset({1, 3, netInputSizes.at(i).height, netInputSizes.at(i).width});
            op::uCharCvMatToFloatPtr(inputNetData[i].getPtr(), OP_CV2OPMAT(frameWithNetSize), 1);

            // // OpenCV equivalent
            // const auto scale = 1/255.;
            // const cv::Scalar mean{128,128,128};
            // const cv::Size outputSize{netInputSizes[i].width, netInputSizes[i].height};
            // // cv::Mat cvMat;
            // cv::dnn::blobFromImage(
            //     // frameWithNetSize, cvMat, scale, outputSize, mean);
            //     frameWithNetSize, inputNetData[i].getCvMat(), scale, outputSize, mean);
            // // op::opLog(cv::norm(cvMat - inputNetData[i].getCvMat())); // ~0.25
        }

        // outputData - Rescale keeping aspect ratio and transform to float the output image
        op::Array<float> outputData({netOutputSize.y, netOutputSize.x, 3}); // This size is used everywhere
        cv::Mat frameWithOutputSize;
        resizeFixedAspectRatio(frameWithOutputSize, cvInputData, scaleInputToOutput, netOutputSize);
        frameWithOutputSize.convertTo(OP_OP2CVMAT(outputData.getCvMat()), CV_32FC3);

        op::Array<float> poseNetOutput;
        op::Array<float> poseKeypoints;
        poseKeypoints = extractPoseKeypoints(inputNetData, inputSize, scaleInputToNetInputs, poseNetOutput);

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
