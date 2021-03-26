#pragma once

#include <c4a/pose/IBodyFromImage.h>

//-----------------------------------------------------------------------------
// op forward declares
//-----------------------------------------------------------------------------
namespace op {
    template<typename T>
    class Array;
    class Matrix;
} // namespace op

namespace c4a {
namespace pose {

//-----------------------------------------------------------------------------
// BodyFromImage
//-----------------------------------------------------------------------------
class BodyFromImage : public IBodyFromImage
{
public:
    //-------------------------------------------------------------------------
    BodyFromImage() = default;

    //-------------------------------------------------------------------------
    ~BodyFromImage() override = default;

    // Execute a class method
    int run(const char* filename) override;

private:
    void display(const op::Matrix& cvOutputData);
    void printKeypoints(const op::Array<float>& poseKeypoints);
};

} // namespace pose
} // namespace c4a
