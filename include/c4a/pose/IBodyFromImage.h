#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

namespace c4a {
namespace pose {

// Forward-declare shared pointer
class IBodyFromImage;
using IBodyFromImagePtr = std::shared_ptr<IBodyFromImage>;

//-----------------------------------------------------------------------------
class IBodyFromImage
{
public:
    // Factory method
    static IBodyFromImagePtr create();

    // Run the algorithm
    virtual int run(const char* filename) = 0;
    virtual int run(cv::Mat image)        = 0;

protected:
    virtual ~IBodyFromImage() { }
};

} // namespace pose
} // namespace c4a
