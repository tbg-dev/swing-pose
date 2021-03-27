#pragma once

#include <c4a/pose/IBodyFromImage.h>

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
};

} // namespace pose
} // namespace c4a
