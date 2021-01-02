import os, platform, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'build', 'python'))
from c4a.conanfile import c4aConanFile

class c4aPoseConan(c4aConanFile):
    name = "c4a-pose"
    version = "0.0.1"
    url = "https://github.com/c4a-dev/pose"
    description = "Pose library built on OpenPose."
    requires = (
        "c4a-core/0.0.1@c4a-skeleton/stable",
        "openpose/1.7.0@c4a-skeleton/stable"
    )