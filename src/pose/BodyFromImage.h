#pragma once

#include <c4a/pose/IBodyFromImage.h>

//-----------------------------------------------------------------------------
// op forward declares
//-----------------------------------------------------------------------------
namespace op {
    struct Datum;
    using DatumPtr = std::shared_ptr<op::Datum>;
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
    void display(const std::shared_ptr<std::vector<op::DatumPtr>>& datumsPtr);
    void printKeypoints(const std::shared_ptr<std::vector<op::DatumPtr>>& datumsPtr);
};

} // namespace pose
} // namespace c4a
