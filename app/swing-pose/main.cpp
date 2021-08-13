#include <iostream>

#include <boost/program_options.hpp>
#include <c4a/core/log.h>
#include <c4a/gst/IPipeline.h>
#include <c4a/pose/IBodyFromImage.h>
#include <opencv2/opencv.hpp>

#define JPEGDEC

static constexpr auto LGR = "main";

// TODO: use synchronized deque
std::deque<cv::Mat> frameQueue;

//-----------------------------------------------------------------------------
// Entry point when not running as a Windows background application.
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    c4a::core::Logging::initConsoleLogging();
    c4a::core::Logging::setLogLevel(c4a::core::Logging::Level::TRACE);

    // Create the pipeline
    auto pipeline = c4a::gst::IPipeline::create("pipeline");

    auto tcpsrc = c4a::gst::IElement::create("tcpclientsrc", "tcpsrc");
    auto queue = c4a::gst::IElement::create("queue", "queue");
    auto appsink = c4a::gst::IElement::create("appsink", "appsink");
    auto jpegdec = c4a::gst::IElement::create("jpegdec", "jpegdec");
    auto videoconvert = c4a::gst::IElement::create("videoconvert", "videoconvert");
    //auto videoscale = c4a::gst::IElement::create("videoscale", "videoscale");
    //auto autovideosink = c4a::gst::IElement::create("autovideosink", "autovideosink");

    GstCaps* bgr = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            nullptr
            );

    tcpsrc
        ->setProp("host", "127.0.0.1") // superfast
         .setProp("port", 4321);

    appsink
        ->setProp("sync", true)
         .setProp("emit-signals", true)
         .onSignal(
                 "new-sample",
                 +[] (GstElement* sink, void *data) {
                     auto element = c4a::gst::IElement::findElement(sink);
                     LOG_DEBUG(LGR, "New Sample! " + std::string(element->getName()));

                     // Retrieve the buffer
                     GstSample* sample;
                     g_signal_emit_by_name(sink, "pull-sample", &sample);
                     if (sample) {
                         GstCaps* caps =  gst_sample_get_caps(sample);
                         if (caps != nullptr) {
                             LOG_DEBUG(LGR, gst_caps_to_string(caps));
                         }

                         const GstStructure* info = gst_sample_get_info(sample);
                         if (info != nullptr) {
                             LOG_DEBUG(LGR, gst_structure_to_string(info));
                         }

                         GstBuffer* buffer = gst_sample_get_buffer(sample);
                         if (buffer != nullptr) {
                             GstMapInfo map;
                             gst_buffer_map(buffer, &map, GST_MAP_READ);

#ifdef JPEGDEC

                             // convert gstreamer data to OpenCV Mat, you could actually
                             // resolve height / width from caps...
                             cv::Mat frame(cv::Size(640, 480), CV_8UC3, (char*)map.data, cv::Mat::AUTO_STEP);

#else

                             // Create a Size(1, nSize) Mat object of 8-bit, single-byte elements
                             cv::Mat rawData( 1, (int) map.size, CV_8UC1, (char*) map.data );
                             cv::Mat frame = cv::imdecode(rawData, cv::IMREAD_COLOR);
                             if (frame.data == nullptr) {
                                 // Error reading raw image data
                                 LOG_ERROR(LGR, "Couldn't decode image");
                             } else {
                                 LOG_INFO(LGR, "Decoded image!");
                             }

#endif

                             // TODO: synchronize this....
                             frameQueue.push_back(frame);

                             gst_buffer_unmap(buffer, &map);
                        }
                        gst_sample_unref(sample);
                        return GST_FLOW_OK;
                    }
                    return GST_FLOW_ERROR;
                },
                nullptr);

    LOG_TRACE(LGR, "Building pipeline");
    pipeline
        ->add(tcpsrc)
         .addAndLink(queue)
 #ifdef JPEGDEC
         .addAndLink(jpegdec)
         .addAndLink(videoconvert)
         .addAndLink(appsink, bgr)
 #else
         .addAndLink(appsink)
 #endif
         //.addAndLink(videoscale)
         //.addAndLink(autovideosink)
         .play();


    auto bodyFromImage = c4a::pose::IBodyFromImage::create();

    //cv::namedWindow("frame",1);
    int key = -1;
    while(key < 0) {
        pipeline->runIteration();

        // TODO: synchronize...
        if (!frameQueue.empty()) {
            // this lags pretty badly even when grabbing frames from webcam
            cv::Mat frame = frameQueue.front();
            //imshow("frame", frame);
            //key = cv::waitKey(30);
            bodyFromImage->run(frame);
            frameQueue.clear();
        }
    }

    //pipeline->waitForEnd();

    return EXIT_SUCCESS;
}