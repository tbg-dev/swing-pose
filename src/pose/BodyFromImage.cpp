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
void addCaffeNetOnThread(
    std::vector<std::shared_ptr<op::Net>>& net,
    std::vector<std::shared_ptr<op::ArrayCpuGpu<float>>>& caffeNetOutputBlob,
    const op::PoseModel poseModel, const int gpuId, const std::string& modelFolder,
    const std::string& protoTxtPath, const std::string& caffeModelPath, const bool enableGoogleLogging)
{
    try
{
    // Add Caffe Net
    net.emplace_back(
            std::make_shared<op::NetCaffe>(
                modelFolder + (protoTxtPath.empty() ? getPoseProtoTxt(poseModel) : protoTxtPath),
                modelFolder + (caffeModelPath.empty() ? getPoseTrainedModel(poseModel) : caffeModelPath),
                gpuId, enableGoogleLogging));
    // net.emplace_back(
    //     std::make_shared<NetOpenCv>(
    //         modelFolder + (protoTxtPath.empty() ? getPoseProtoTxt(poseModel) : protoTxtPath),
    //         modelFolder + (caffeModelPath.empty() ? getPoseTrainedModel(poseModel) : caffeModelPath),
    //         gpuId));
    // UNUSED(enableGoogleLogging);
    // Initializing them on the thread
    net.back()->initializationOnThread();
    caffeNetOutputBlob.emplace_back((net.back().get())->getOutputBlobArray());
    // Sanity check
    if (net.size() != caffeNetOutputBlob.size())
        op::error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
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
inline void reshapePoseExtractorCaffe(
    std::shared_ptr<op::ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
    std::shared_ptr<op::NmsCaffe<float>>& nmsCaffe,
    std::shared_ptr<op::BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
    std::shared_ptr<op::MaximumCaffe<float>>& maximumCaffe,
    std::vector<std::shared_ptr<op::ArrayCpuGpu<float>>>& caffeNetOutputBlobsShared,
    std::shared_ptr<op::ArrayCpuGpu<float>>& heatMapsBlob, std::shared_ptr<op::ArrayCpuGpu<float>>& peaksBlob,
    std::shared_ptr<op::ArrayCpuGpu<float>>& maximumPeaksBlob, const float scaleInputToNetInput,
    const op::PoseModel poseModel, const int gpuId, const float upsamplingRatio, const bool topDownRefinement)
{
    try
    {
        const auto netDescreaseFactor = (
                upsamplingRatio <= 0.f ? getPoseNetDecreaseFactor(poseModel) : upsamplingRatio);
        // HeatMaps extractor blob and layer
        // Caffe modifies bottom - Heatmap gets resized
        const auto caffeNetOutputBlobs = arraySharedToPtr(caffeNetOutputBlobsShared);
        resizeAndMergeCaffe->Reshape(
                caffeNetOutputBlobs, {heatMapsBlob.get()},
        netDescreaseFactor, 1.f/scaleInputToNetInput, true, gpuId);
        // Pose extractor blob and layer
        nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, op::getPoseMaxPeaks(),
            getPoseNumberBodyParts(poseModel), gpuId);
        // Pose extractor blob and layer
        bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()}, gpuId);
        if (topDownRefinement)
            maximumCaffe->Reshape({heatMapsBlob.get()}, {maximumPeaksBlob.get()});
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

//-----------------------------------------------------------------------------
static op::Array<float> extractPoseKeypoints(
    const op::PoseModel poseModel, const std::string& modelFolder, const int gpuId,
    const std::vector<op::HeatMapType>& heatMapTypes, const op::ScaleMode heatMapScaleMode,
    const bool maximizePositives, const std::string& protoTxtPath, const std::string& caffeModelPath,
    const float upsamplingRatio, const bool enableNet, const bool enableGoogleLogging,
    const std::vector<op::Array<float>>& inputNetData, const cv::Size& inputDataSize,
    const std::vector<double>& scaleInputToNetInputs, const op::Array<float>& poseNetOutput)
{
    const bool TOP_DOWN_REFINEMENT = false; // Note: +5% acc 1 scale, -2% max acc setting

    cv::Size mNetOutputSize{0,0};
    op::Array<float> mPoseKeypoints;
    op::Array<float> mPoseScores;
    float mScaleNetToOutput;
    std::array<std::atomic<double>, (int)op::PoseProperty::Size> mProperties;
    std::thread::id mThreadId;
    {
        try
        {
            // Error check
            if (heatMapScaleMode != op::ScaleMode::ZeroToOne
                && heatMapScaleMode != op::ScaleMode::ZeroToOneFixedAspect
                && heatMapScaleMode != op::ScaleMode::PlusMinusOne
                && heatMapScaleMode != op::ScaleMode::PlusMinusOneFixedAspect
                && heatMapScaleMode != op::ScaleMode::UnsignedChar && heatMapScaleMode != op::ScaleMode::NoScale)
                op::error("The ScaleMode heatMapScaleMode must be ZeroToOne, ZeroToOneFixedAspect, PlusMinusOne,"
                      " PlusMinusOneFixedAspect or UnsignedChar.", __LINE__, __FUNCTION__, __FILE__);

            // Properties - Init to 0
            for (auto& property : mProperties)
                property = 0.;
            // Properties - Fill default values
            mProperties[(int)op::PoseProperty::NMSThreshold] = getPoseDefaultNmsThreshold(poseModel, maximizePositives);
            mProperties[(int)op::PoseProperty::ConnectInterMinAboveThreshold]
                    = op::getPoseDefaultConnectInterMinAboveThreshold(maximizePositives);
            mProperties[(int)op::PoseProperty::ConnectInterThreshold] = getPoseDefaultConnectInterThreshold(
                    poseModel, maximizePositives);
            mProperties[(int)op::PoseProperty::ConnectMinSubsetCnt] = op::getPoseDefaultMinSubsetCnt(maximizePositives);
            mProperties[(int)op::PoseProperty::ConnectMinSubsetScore] = op::getPoseDefaultConnectMinSubsetScore(
                    maximizePositives);
        }
        catch (const std::exception& e)
        {
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // General parameters
    std::vector<std::shared_ptr<op::Net>> spNets;
    std::shared_ptr<op::ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe{std::make_shared<op::ResizeAndMergeCaffe<float>>()};
    std::shared_ptr<op::NmsCaffe<float>> spNmsCaffe{std::make_shared<op::NmsCaffe<float>>()};
    std::shared_ptr<op::BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe{std::make_shared<op::BodyPartConnectorCaffe<float>>()};
    std::shared_ptr<op::MaximumCaffe<float>> spMaximumCaffe{(TOP_DOWN_REFINEMENT ? std::make_shared<op::MaximumCaffe<float>>() : nullptr)};
    std::vector<std::vector<int>> mNetInput4DSizes;
    // Init with thread
    std::vector<std::shared_ptr<op::ArrayCpuGpu<float>>> spCaffeNetOutputBlobs;
    std::shared_ptr<op::ArrayCpuGpu<float>> spHeatMapsBlob;
    std::shared_ptr<op::ArrayCpuGpu<float>> spPeaksBlob;
    std::shared_ptr<op::ArrayCpuGpu<float>> spMaximumPeaksBlob;
    {
        try
        {
            // Layers parameters
            spBodyPartConnectorCaffe->setPoseModel(poseModel);
            spBodyPartConnectorCaffe->setMaximizePositives(maximizePositives);
        }
        catch (const std::exception& e)
        {
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const int mNumberPeopleMax{-1};
    const int mTracking{-1};
    const std::shared_ptr<op::KeepTopNPeople> spKeepTopNPeople{nullptr};
    const std::shared_ptr<op::PersonIdExtractor> spPersonIdExtractor{nullptr};
    const std::shared_ptr<std::vector<std::shared_ptr<op::PersonTracker>>> spPersonTrackers;

    // Get thread id
    mThreadId = {std::this_thread::get_id()};
    // Deep net initialization
    {
        try
        {
            if (enableNet)
                {
                    // Logging
                    op::opLog("Starting initialization on thread.", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // Initialize Caffe net
                    addCaffeNetOnThread(
                        spNets, spCaffeNetOutputBlobs, poseModel, gpuId,
                        modelFolder, protoTxtPath, caffeModelPath,
                        enableGoogleLogging);
                }
                // Initialize blobs
                spHeatMapsBlob = {std::make_shared<op::ArrayCpuGpu<float>>(1,1,1,1)};
                spPeaksBlob = {std::make_shared<op::ArrayCpuGpu<float>>(1,1,1,1)};
                if (TOP_DOWN_REFINEMENT)
                    spMaximumPeaksBlob = {std::make_shared<op::ArrayCpuGpu<float>>(1,1,1,1)};
                // Logging
                op::opLog("Finished initialization on thread.", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // forwardPass
    {
        try
        {
            // const auto REPS = 1;
            // double timeNormalize1 = 0.;
            // double timeNormalize2 = 0.;
            // double timeNormalize3 = 0.;
            // double timeNormalize4 = 0.;
            // OP_CUDA_PROFILE_INIT(REPS);
            // Sanity checks
            if (inputNetData.empty())
                op::error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
            for (const auto& inputNetDataI : inputNetData)
                if (inputNetDataI.empty())
                    op::error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
            if (inputNetData.size() != scaleInputToNetInputs.size())
                op::error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                      __LINE__, __FUNCTION__, __FILE__);
            if (poseNetOutput.empty() != enableNet)
            {
                const std::string errorMsg = ". Either use OpenPose default network (`--body 1`) or fill the"
                    " `poseNetOutput` argument (only 1 of those 2, not both).";
                if (poseNetOutput.empty())
                    op::error("The argument poseNetOutput cannot be empty if mEnableNet is true" + errorMsg,
                          __LINE__, __FUNCTION__, __FILE__);
                else
                    op::error("The argument poseNetOutput is not empty and you have also explicitly chosen to run"
                          " the OpenPose network" + errorMsg, __LINE__, __FUNCTION__, __FILE__);
            }

            // Resize std::vectors if required
            const auto numberScales = inputNetData.size();
            mNetInput4DSizes.resize(numberScales);

            // Process each image - Caffe deep network
            if (enableNet)
            {
                while (spNets.size() < numberScales)
                    addCaffeNetOnThread(
                        spNets, spCaffeNetOutputBlobs, poseModel, gpuId,
                        modelFolder, protoTxtPath, caffeModelPath, false);

                for (auto i = 0u ; i < inputNetData.size(); i++)
                    spNets.at(i)->forwardPass(inputNetData[i]);
            }
            // If custom network output
            else
            {
                // Sanity check
                if (inputNetData.size() != 1u)
                    op::error("Size(inputNetData) must match the provided heatmaps batch size ("
                          + std::to_string(inputNetData.size()) + " vs. " + std::to_string(1) + ").",
                          __LINE__, __FUNCTION__, __FILE__);
                // Copy heatmap information
                spCaffeNetOutputBlobs.clear();
                const bool copyFromGpu = false;
                spCaffeNetOutputBlobs.emplace_back(
                    std::make_shared<op::ArrayCpuGpu<float>>(poseNetOutput, copyFromGpu));
            }
            // Reshape blobs if required
            for (auto i = 0u ; i < inputNetData.size(); i++)
            {
                // Reshape blobs if required - For dynamic sizes (e.g., images of different aspect ratio)
                const auto changedVectors = !op::vectorsAreEqual(
                    mNetInput4DSizes.at(i), inputNetData[i].getSize());
                if (changedVectors)
                {
                    mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                    reshapePoseExtractorCaffe(
                        spResizeAndMergeCaffe, spNmsCaffe, spBodyPartConnectorCaffe,
                        spMaximumCaffe, spCaffeNetOutputBlobs, spHeatMapsBlob,
                        spPeaksBlob, spMaximumPeaksBlob, 1.f, poseModel,
                        gpuId, upsamplingRatio, TOP_DOWN_REFINEMENT);
                        // In order to resize to input size to have same results as Matlab
                        // scaleInputToNetInputs[i] vs. 1.f
                }
                // Get scale net to output (i.e., image input)
                const auto ratio = (
                    upsamplingRatio <= 0.f
                        ? 1 : upsamplingRatio / getPoseNetDecreaseFactor(poseModel));
                if (changedVectors || TOP_DOWN_REFINEMENT)
                    mNetOutputSize = cv::Size{
                        positiveIntRound(ratio*mNetInput4DSizes[0][3]),
                        positiveIntRound(ratio*mNetInput4DSizes[0][2])};
            }
            // OP_CUDA_PROFILE_END(timeNormalize1, 1e3, REPS);
            // OP_CUDA_PROFILE_INIT(REPS);
            // 2. Resize heat maps + merge different scales
            // ~5ms (GPU) / ~20ms (CPU)
            const auto caffeNetOutputBlobs = arraySharedToPtr(spCaffeNetOutputBlobs);
            // Set and fill floatScaleRatios
                // Option 1/2 (warning for double-to-float conversion)
            // const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                // Option 2/2
            std::vector<float> floatScaleRatios;
            std::for_each(
                scaleInputToNetInputs.begin(), scaleInputToNetInputs.end(),
                [&floatScaleRatios](const double value) { floatScaleRatios.emplace_back(float(value)); });
            spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
            spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {spHeatMapsBlob.get()});
            // Get scale net to output (i.e., image input)
            // Note: In order to resize to input size, (un)comment the following lines
            const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
            const cv::Size netSize{
                positiveIntRound(scaleProducerToNetInput*inputDataSize.width),
                positiveIntRound(scaleProducerToNetInput*inputDataSize.height)};
            mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
            // mScaleNetToOutput = 1.f;
            // 3. Get peaks by Non-Maximum Suppression
            // ~2ms (GPU) / ~7ms (CPU)
            // OP_CUDA_PROFILE_END(timeNormalize2, 1e3, REPS);
            const auto nmsThreshold = (float)mProperties.at((int)op::PoseProperty::NMSThreshold);
            const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
            // OP_CUDA_PROFILE_INIT(REPS);
            spNmsCaffe->setThreshold(nmsThreshold);
            spNmsCaffe->setOffset(op::Point<float>{nmsOffset, nmsOffset});
            spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
            // 4. Connecting body parts
            // OP_CUDA_PROFILE_END(timeNormalize3, 1e3, REPS);
            // OP_CUDA_PROFILE_INIT(REPS);
            spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
            spBodyPartConnectorCaffe->setDefaultNmsThreshold((float)mProperties.at((int)op::PoseProperty::NMSThreshold));
            spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                (float)mProperties.at((int)op::PoseProperty::ConnectInterMinAboveThreshold));
            spBodyPartConnectorCaffe->setInterThreshold((float)mProperties.at((int)op::PoseProperty::ConnectInterThreshold));
            spBodyPartConnectorCaffe->setMinSubsetCnt((int)mProperties.at((int)op::PoseProperty::ConnectMinSubsetCnt));
            spBodyPartConnectorCaffe->setMinSubsetScore((float)mProperties.at((int)op::PoseProperty::ConnectMinSubsetScore));
            // Note: BODY_25D will crash (only implemented for CPU version)
            spBodyPartConnectorCaffe->Forward(
                {spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);
            // OP_CUDA_PROFILE_END(timeNormalize4, 1e3, REPS);
            // opLog("1(caf)= " + std::to_string(timeNormalize1) + "ms");
            // opLog("2(res) = " + std::to_string(timeNormalize2) + " ms");
            // opLog("3(nms) = " + std::to_string(timeNormalize3) + " ms");
            // opLog("4(bpp) = " + std::to_string(timeNormalize4) + " ms");
            // Re-run on each person
            if (TOP_DOWN_REFINEMENT)
            {
                // Get each person rectangle
                for (auto person = 0 ; person < mPoseKeypoints.getSize(0) ; person++)
                {
                    // Get person rectangle resized to input size
                    const auto rectangleF = getKeypointsRectangle(mPoseKeypoints, person, nmsThreshold)
                                          / mScaleNetToOutput;
                    // Make rectangle bigger to make sure the whole body is inside
                    op::Rectangle<int> rectangleInt{
                        positiveIntRound(rectangleF.x - 0.2*rectangleF.width),
                        positiveIntRound(rectangleF.y - 0.2*rectangleF.height),
                        positiveIntRound(rectangleF.width*1.4),
                        positiveIntRound(rectangleF.height*1.4)
                    };
                    keepRoiInside(rectangleInt, inputNetData[0].getSize(3), inputNetData[0].getSize(2));
                    // Input size
                    // // Note: In order to preserve speed but maximize accuracy
                    // // If e.g. rectange = 10x1 and inputSize = 656x368 --> targetSize = 656x368
                    // // Note: If e.g. rectange = 1x10 and inputSize = 656x368 --> targetSize = 368x656
                    // const auto width = ( ? rectangleInt.width : rectangleInt.height);
                    // const auto height = (width == rectangleInt.width ? rectangleInt.height : rectangleInt.width);
                    // const Point<int> inputSize{width, height};
                    // Note: If inputNetData.size = -1x368 --> TargetSize = 368x-1
                    const cv::Point inputSizeInit{rectangleInt.width, rectangleInt.height};
                    // Target size
                    cv::Point targetSize;
                    // Optimal case (using training size)
                    if (inputNetData[0].getSize(2) >= 368 || inputNetData[0].getVolume(2,3) >= 135424) // 368^2
                        targetSize = cv::Point{368, 368};
                    // Low resolution cases: Keep same area than biggest scale
                    else
                    {
                        const auto minSide = fastMin(
                            368, fastMin(inputNetData[0].getSize(2), inputNetData[0].getSize(3)));
                        const auto maxSide = fastMin(
                            368, fastMax(inputNetData[0].getSize(2), inputNetData[0].getSize(3)));
                        // Person bounding box is vertical
                        if (rectangleInt.width < rectangleInt.height)
                            targetSize = cv::Point{minSide, maxSide};
                        // Person bounding box is horizontal
                        else
                            targetSize = cv::Point{maxSide, minSide};
                    }
                    // Fill resizedImage
                    /*const*/ auto scaleNetToRoi = resizeGetScaleFactor(inputSizeInit, targetSize);
                    // Update rectangle to avoid black padding and instead take full advantage of the network area
                    const auto padding = op::Point<int>{
                        (int)std::round((targetSize.x-1) / scaleNetToRoi + 1 - inputSizeInit.x),
                        (int)std::round((targetSize.y-1) / scaleNetToRoi + 1 - inputSizeInit.y)
                    };
                    // Width requires padding
                    if (padding.x > 2 || padding.y > 2) // 2 pixels as threshold
                    {
                        if (padding.x > 2) // 2 pixels as threshold
                        {
                            rectangleInt.x -= padding.x/2;
                            rectangleInt.width += padding.x;
                        }
                        else if (padding.y > 2) // 2 pixels as threshold
                        {
                            rectangleInt.y -= padding.y/2;
                            rectangleInt.height += padding.y;
                        }
                        keepRoiInside(rectangleInt, inputNetData[0].getSize(3), inputNetData[0].getSize(2));
                        scaleNetToRoi = resizeGetScaleFactor(
                            cv::Point{rectangleInt.width, rectangleInt.height}, targetSize);
                    }
                    // No if scaleNetToRoi < 1 (image would be shrinked, so we assume best result already obtained)
                    if (scaleNetToRoi > 1)
                    {
                        const auto areaInput = inputNetData[0].getVolume(2,3);
                        const auto areaRoi = targetSize.x * targetSize.y;
                        op::Array<float> inputNetDataRoi{{1, 3, targetSize.y, targetSize.x}};
                        for (auto c = 0u ; c < 3u ; c++)
                        {
                            // Input image
                            const cv::Mat wholeInputCvMat(
                                inputNetData[0].getSize(2), inputNetData[0].getSize(3), CV_32FC1,
                                inputNetData[0].getPseudoConstPtr() + c * areaInput);
                            // Input image cropped
                            const cv::Mat inputCvMat(
                                wholeInputCvMat, cv::Rect{rectangleInt.x, rectangleInt.y, rectangleInt.width, rectangleInt.height});
                            // Resize image for inputNetDataRoi
                            cv::Mat resizedImageCvMat(
                                inputNetDataRoi.getSize(2), inputNetDataRoi.getSize(3), CV_32FC1,
                                inputNetDataRoi.getPtr() + c * areaRoi);
                            resizeFixedAspectRatio(resizedImageCvMat, inputCvMat, scaleNetToRoi, targetSize);
                        }

                        // Re-Process image
                        // 1. Caffe deep network
                        spNets.at(0)->forwardPass(inputNetDataRoi);
                        std::vector<std::shared_ptr<op::ArrayCpuGpu<float>>> caffeNetOutputBlob{
                            spCaffeNetOutputBlobs[0]};
                        // Reshape blobs
                        if (!op::vectorsAreEqual(mNetInput4DSizes.at(0), inputNetDataRoi.getSize()))
                        {
                            mNetInput4DSizes.at(0) = inputNetDataRoi.getSize();
                            reshapePoseExtractorCaffe(
                                spResizeAndMergeCaffe, spNmsCaffe,
                                spBodyPartConnectorCaffe, spMaximumCaffe,
                                // spCaffeNetOutputBlobs,
                                caffeNetOutputBlob, spHeatMapsBlob, spPeaksBlob,
                                spMaximumPeaksBlob, 1.f, poseModel, gpuId,
                                upsamplingRatio, TOP_DOWN_REFINEMENT);
                        }
                        // 2. Resize heat maps + merge different scales
                        const auto caffeNetOutputBlobsNew = arraySharedToPtr(caffeNetOutputBlob);
                        // const std::vector<float> floatScaleRatiosNew(
                        //     scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                        const std::vector<float> floatScaleRatiosNew{(float)scaleInputToNetInputs[0]};
                        spResizeAndMergeCaffe->setScaleRatios(floatScaleRatiosNew);
                        spResizeAndMergeCaffe->Forward(
                            caffeNetOutputBlobsNew, {spHeatMapsBlob.get()});
                        // Get scale net to output (i.e., image input)
                        const auto scaleRoiToOutput = float(mScaleNetToOutput / scaleNetToRoi);
                        // 3. Get peaks by Non-Maximum Suppression
                        const auto nmsThresholdRefined = 0.02f;
                        spNmsCaffe->setThreshold(nmsThresholdRefined);
                        const auto nmsOffsetNew = float(0.5/double(scaleRoiToOutput));
                        spNmsCaffe->setOffset(op::Point<float>{nmsOffsetNew, nmsOffsetNew});
                        spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                        // Define poseKeypoints
                        op::Array<float> poseKeypoints;
                        op::Array<float> poseScores;
                        // 4. Connecting body parts
                        // Get scale net to output (i.e., image input)
                        spBodyPartConnectorCaffe->setScaleNetToOutput(scaleRoiToOutput);
                        spBodyPartConnectorCaffe->setInterThreshold(0.01f);
                        spBodyPartConnectorCaffe->Forward(
                            {spHeatMapsBlob.get(), spPeaksBlob.get()}, poseKeypoints, poseScores);
                        // If detected people in new subnet
                        if (!poseKeypoints.empty())
                        {
                            // // Scale back keypoints
                            const auto xOffset = float(rectangleInt.x*mScaleNetToOutput);
                            const auto yOffset = float(rectangleInt.y*mScaleNetToOutput);
                            scaleKeypoints2d(poseKeypoints, 1.f, 1.f, xOffset, yOffset);
                            // Re-assign person back
                            // // Option a) Just use biggest person (simplest but fails with crowded people)
                            // const auto personRefined = getBiggestPerson(poseKeypoints, nmsThreshold);
                            // Option b) Get minimum keypoint distance
                            // Get min distance
                            int personRefined = -1;
                            float personAverageDistance = std::numeric_limits<float>::max();
                            for (auto person2 = 0 ; person2 < poseKeypoints.getSize(0) ; person2++)
                            {
                                // Get average distance
                                const auto currentAverageDistance = getDistanceAverage(
                                    mPoseKeypoints, person, poseKeypoints, person2, nmsThreshold);
                                // Update person
                                if (personAverageDistance > currentAverageDistance
                                    && getNonZeroKeypoints(poseKeypoints, person2, nmsThreshold)
                                        >= 0.75*getNonZeroKeypoints(mPoseKeypoints, person, nmsThreshold))
                                {
                                    personRefined = person2;
                                    personAverageDistance = currentAverageDistance;
                                }
                            }
                            // Get max ROI
                            int personRefinedRoi = -1;
                            float personRoi = -1.f;
                            for (auto person2 = 0 ; person2 < poseKeypoints.getSize(0) ; person2++)
                            {
                                // Get ROI
                                const auto currentRoi = getKeypointsRoi(
                                    mPoseKeypoints, person, poseKeypoints, person2, nmsThreshold);
                                // Update person
                                if (personRoi < currentRoi
                                    && getNonZeroKeypoints(poseKeypoints, person2, nmsThreshold)
                                        >= 0.75*getNonZeroKeypoints(mPoseKeypoints, person, nmsThreshold))
                                {
                                    personRefinedRoi = person2;
                                    personRoi = currentRoi;
                                }
                            }
                            // If good refined candidate found
                            // I.e., if both max ROI and min dist match on same person id
                            if (personRefined == personRefinedRoi && personRefined > -1)
                            {
                                // Update only if avg dist is small enough
                                const auto personRectangle = getKeypointsRectangle(
                                    mPoseKeypoints, person, nmsThreshold);
                                const auto personRatio = 0.1f * (float)std::sqrt(
                                    personRectangle.x*personRectangle.x + personRectangle.y*personRectangle.y);
                                // if (mPoseScores[person] < poseScores[personRefined]) // This harms accuracy
                                if (personAverageDistance < personRatio)
                                {
                                    const auto personArea = mPoseKeypoints.getVolume(1,2);
                                    const auto personIndex = person * personArea;
                                    const auto personRefinedIndex = personRefined * personArea;
                                    // mPoseKeypoints: Update keypoints
                                    // Option a) Using refined ones
                                    std::copy(
                                        poseKeypoints.getPtr() + personRefinedIndex,
                                        poseKeypoints.getPtr() + personRefinedIndex + personArea,
                                        mPoseKeypoints.getPtr() + personIndex);
                                    mPoseScores[person] = poseScores[personRefined];
                                    // // Option b) Using ones with highest score (-6% acc single scale)
                                    // // Fill gaps
                                    // for (auto part = 0 ; part < mPoseKeypoints.getSize(1) ; part++)
                                    // {
                                    //     // For currently empty keypoints
                                    //     const auto partIndex = personIndex+3*part;
                                    //     const auto partRefinedIndex = personRefinedIndex+3*part;
                                    //     const auto scoreDifference = poseKeypoints[partRefinedIndex+2]
                                    //                                - mPoseKeypoints[partIndex+2];
                                    //     if (scoreDifference > 0)
                                    //     {
                                    //         const auto x = poseKeypoints[partRefinedIndex];
                                    //         const auto y = poseKeypoints[partRefinedIndex + 1];
                                    //         mPoseKeypoints[partIndex] = x;
                                    //         mPoseKeypoints[partIndex+1] = y;
                                    //         mPoseKeypoints[partIndex+2] += scoreDifference;
                                    //         mPoseScores[person] += scoreDifference;
                                    //     }
                                    // }

                                    // No acc improvement (-0.05% acc single scale)
                                    // // Finding all missing peaks (CPM-style)
                                    // // Only if no other person in there (otherwise 2% accuracy drop)
                                    // if (getNonZeroKeypoints(mPoseKeypoints, person, nmsThresholdRefined) > 0)
                                    // {
                                    //     // Get whether 0% ROI with other people
                                    //     // Get max ROI
                                    //     bool overlappingPerson = false;
                                    //     for (auto person2 = 0 ; person2 < mPoseKeypoints.getSize(0) ; person2++)
                                    //     {
                                    //         if (person != person2)
                                    //         {
                                    //             // Get ROI
                                    //             const auto currentRoi = getKeypointsRoi(
                                    //                 mPoseKeypoints, person, person2, nmsThreshold);
                                    //             // Update person
                                    //             if (currentRoi > 0.f)
                                    //             {
                                    //                 overlappingPerson = true;
                                    //                 break;
                                    //             }
                                    //         }
                                    //     }
                                    //     if (!overlappingPerson)
                                    //     {
                                    //         // Get keypoint with maximum probability per channel
                                    //         spMaximumCaffe->Forward(
                                    //             {spHeatMapsBlob.get()}, {spMaximumPeaksBlob.get()});
                                    //         // Fill gaps
                                    //         const auto* posePeaksPtr = spMaximumPeaksBlob->mutable_cpu_data();
                                    //         for (auto part = 0 ; part < mPoseKeypoints.getSize(1) ; part++)
                                    //         {
                                    //             // For currently empty keypoints
                                    //             if (mPoseKeypoints[personIndex+3*part+2] < nmsThresholdRefined)
                                    //             {
                                    //                 const auto xyIndex = 3*part;
                                    //                 const auto x = posePeaksPtr[xyIndex]*scaleRoiToOutput + xOffset;
                                    //                 const auto y = posePeaksPtr[xyIndex + 1]*scaleRoiToOutput + yOffset;
                                    //                 const auto rectangle = getKeypointsRectangle(
                                    //                     mPoseKeypoints, person, nmsThresholdRefined);
                                    //                 if (x >= rectangle.x && x < rectangle.x + rectangle.width
                                    //                     && y >= rectangle.y && y < rectangle.y + rectangle.height)
                                    //                 {
                                    //                     const auto score = posePeaksPtr[xyIndex + 2];
                                    //                     const auto baseIndex = personIndex + 3*part;
                                    //                     mPoseKeypoints[baseIndex] = x;
                                    //                     mPoseKeypoints[baseIndex+1] = y;
                                    //                     mPoseKeypoints[baseIndex+2] = score;
                                    //                     mPoseScores[person] += score;
                                    //                 }
                                    //             }
                                    //         }
                                    //     }
                                    // }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    return mPoseKeypoints.clone();
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
        {
            const op::WrapperStructPose wrapperStructPose;
            const auto modelFolder = op::formatAsDirectory(wrapperStructPose.modelFolder.getStdString());
            poseKeypoints = extractPoseKeypoints(
                    wrapperStructPose.poseModel, modelFolder, wrapperStructPose.gpuNumberStart,
                    wrapperStructPose.heatMapTypes, wrapperStructPose.heatMapScaleMode,
                    wrapperStructPose.maximizePositives,
                    wrapperStructPose.protoTxtPath.getStdString(),
                    wrapperStructPose.caffeModelPath.getStdString(),
                    wrapperStructPose.upsamplingRatio, wrapperStructPose.poseMode == op::PoseMode::Enabled,
                    wrapperStructPose.enableGoogleLogging,
                    inputNetData, inputSize, scaleInputToNetInputs, poseNetOutput);
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
