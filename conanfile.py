from build.python.conanfile import c4aConanFile

class c4aPoseConan(c4aConanFile):
    name = "c4a-pose"
    version = "0.0.1"
    url = "https://github.com/c4a-dev/pose"
    description = "Pose library built on OpenPose."
    requires = (
        "c4a_core/0.0.1@tbg-dev/stable",
        "openpose/1.7.0@tbg-dev/stable",
        "boost/1.75.0",
        "openblas/0.3.13",
        "protobuf/3.9.1"
    )