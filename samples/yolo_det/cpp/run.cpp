#include <memx/mxutils/gui_view.h>
#include <pipeline.h>
#include <signal.h>

#include "memx/accl/MxAccl.h"

#include <chrono>
#include <iostream>
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/opencv.hpp>    /* imshow */
#include <thread>

namespace fs = std::filesystem;
using namespace MX::Pipe;

std::atomic_bool runflag;  // Atomic flag to control run state

// Default model
fs::path model_path = "models/yolov8n/YOLO_v8_nano_640_640_3_onnx.dfp";
// Default post-processing model path
fs::path postprocessing_model_path = "models/yolov8n/YOLO_v8_nano_640_640_3_onnx_post.onnx";

MxQt* gui = nullptr;

#define AVG_FPS_CALC_FRAME_COUNT 50  // Number of frames used to calculate average FPS
#define FRAME_QUEUE_MAX_LENGTH 5

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal) {
    runflag.store(false);  // Stop the program
}

// Function to display usage information
void printUsage(const std::string& programName) {
    std::cout
            << "Usage: " << programName
            << " [-d <dfp_path>] [--video_paths \"cam:0,vid:video_path\"]\n"
            << "Options:\n"
            << "  -d, --dfp_path        (Optional) Path to the DFP. Default: " << model_path
            << postprocessing_model_path << "\n"
            << "  --video_paths         (Optional) Video paths in the format \"cam:0,vid:video_path,vid:video2_path\". Default: cam:0\n";
}

// Function to configure camera settings (resolution and FPS)
bool configureCamera(cv::VideoCapture& vcap) {
    bool settings_success = true;
    try {
        // Attempt to set 640x480 resolution and 30 FPS
        if (!vcap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) ||
            !vcap.set(cv::CAP_PROP_FRAME_WIDTH, 640) || !vcap.set(cv::CAP_PROP_FPS, 30)) {
            std::cout << "Setting vcap Failed\n";
            cv::Mat simpleframe;
            if (!vcap.read(simpleframe)) {
                settings_success = false;
            }
        }
    } catch (...) {
        std::cout << "Exception occurred while setting properties\n";
        settings_success = false;
    }
    return settings_success;
}

// Function to open the camera and apply settings, if not possible, reopen with default settings
bool openCamera(cv::VideoCapture& vcap, int device, int api) {
    vcap.open(device, api);  // Open the camera
    if (!vcap.isOpened()) {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configureCamera(vcap)) {  // Try applying custom settings
        vcap.release();            // Release and reopen with default settings
        vcap.open(device, api);
        if (vcap.isOpened()) {
            std::cout << "Reopened vcap with original resolution\n";
        } else {
            std::cerr << "Failed to reopen vcap\n";
            return false;
        }
    }
    return true;
}

void initVcap(cv::VideoCapture& vcap, const std::string& video_src, bool& src_is_cam) {
    // Open the camera or video source
    if (video_src.substr(0, 3) == "cam") {
        src_is_cam = true;
        int device = std::stoi(video_src.substr(4));
#ifdef __linux__
        if (!openCamera(vcap, device, cv::CAP_V4L2)) {
            throw(std::runtime_error("Failed to open: " + video_src));
        }
#elif defined(_WIN32)
        if (!openCamera(vcap, device, cv::CAP_ANY)) {
            throw(std::runtime_error("Failed to open: " + video_src));
        }
#endif
    } else if (video_src.substr(0, 3) == "vid") {
        src_is_cam = false;
        std::cout << "Video source given = " << video_src.substr(4) << "\n";
        vcap.open(video_src.substr(4), cv::CAP_ANY);
    } else {
        throw(std::runtime_error("Given video src: " + video_src + " is invalid"));
    }

    if (!vcap.isOpened()) {
        std::cout << "videocapture for " << video_src << " is NOT opened\n";
        throw(std::runtime_error("Failed to open: " + video_src));
    }
}

class YoloApp {
  private:
    MX::Pipe::Pipeline* pipe_;

    // Application Variables
    std::deque<cv::Mat> frames_queue;  // Queue for frames
    std::mutex frame_queue_mutex;      // Mutex to control access to the queue
    cv::VideoCapture vcap;             // Video capture object
    bool src_is_cam = false;
    std::vector<float*> ofmaps;  // Buffer for the output of the accelerator
    int length;

    // FPS related
    int num_frames = 0;
    int frame_count = 0;
    float fps_number = .0;  // FPS counter
    std::chrono::milliseconds start_ms;
    std::vector<float> history_fps;

    // Input callback function to fetch frames and preprocess them
    bool in_callback(std::vector<const MX::Types::FeatureMap*> dst, int stream_id) {
        if (runflag.load()) {
            cv::Mat inframe;
            cv::Mat rgbImage;

            while (true) {
                bool got_frame = vcap.read(inframe);  // Capture frame

                if (!got_frame) {  // If no frame, stop the stream
                    std::cout << "No frame \n\n\n";
                    vcap.release();
                    return false;
                }

                if (src_is_cam && (frames_queue.size() >= FRAME_QUEUE_MAX_LENGTH)) {
                    // drop the frame and try again if we've hit the limit
                    continue;
                } else {
                    // Convert frame to RGB and store in queue
                    cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                    {
                        std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                        frames_queue.push_back(rgbImage);
                    }
                }

                // Preprocess
                cv::Mat pre = pipe_->preprocess(rgbImage);
                dst[0]->set_data((float*)pre.data);
                return true;
            }
        } else {
            vcap.release();
            return false;
        }
    }

    // Output callback function to process MXA output and display results
    bool out_callback(std::vector<const MX::Types::FeatureMap*> mxa_outputs, int stream_id) {

        // Get the output data from MXA
        for (int i = 0; i < mxa_outputs.size(); i++) {
            mxa_outputs[i]->get_data(ofmaps[i]);
        }

        // Get the next frame from the queue for display
        cv::Mat displayImage;
        {
            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
            displayImage = frames_queue.front();
            frames_queue.pop_front();
        }

        // Postprocess
        MX::Pipe::Result result;
        pipe_->postprocess(ofmaps, result);
        pipe_->draw(displayImage, result);

        // Display the updated image in the GUI
        if (gui)
            gui->screens[0]->SetDisplayFrame(stream_id, displayImage, fps_number);

        // update fps
        _update_fps(stream_id);

        return true;
    }

    void _update_fps(int stream_id) {
        frame_count++;
        if (frame_count == 1) {
            start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch());
        } else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0) {
            std::chrono::milliseconds duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()) -
                    start_ms;
            fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());

            // Store FPS in history and print
            history_fps.push_back(fps_number);
            std::cout << "Stream " << stream_id << " FPS: " << fps_number << "\n";

            // Reset for next calculation
            frame_count = 0;
        }
    }

  public:
    // Constructor
    YoloApp(MX::Runtime::MxAccl* accl, std::string video_src, int stream_id) {

        // Initialize video capture
        initVcap(vcap, video_src, src_is_cam);

        // init pipeline object
        YoloConfig config;
        config.ori_width = (int)vcap.get(cv::CAP_PROP_FRAME_WIDTH);
        config.ori_height = (int)vcap.get(cv::CAP_PROP_FRAME_HEIGHT);
        config.conf = 0.3f;
        config.iou = 0.4f;
        config.valid_classes = {0};
        pipe_ = MX::Pipe::Pipeline::create("yolov8_detect", config);

        // Get model info and allocate output buffer
        MX::Types::MxModelInfo model_info = accl->get_model_info(0);

        // Allocate memory for MXA outputs
        ofmaps.resize(model_info.num_out_featuremaps);
        for (int i = 0; i < model_info.num_out_featuremaps; i++) {
            ofmaps[i] = new float[model_info.out_featuremap_sizes[i]];
        }

        // Bind input/output callback functions
        auto in_cb = std::bind(
                &YoloApp::in_callback, this, std::placeholders::_1, std::placeholders::_2);
        auto out_cb = std::bind(
                &YoloApp::out_callback, this, std::placeholders::_1, std::placeholders::_2);

        // Connect streams to the accelerator
        accl->connect_stream(in_cb, out_cb, stream_id /* Unique stream id */, 0 /* model idx */);

        // Start the input/output streams
        runflag.store(true);
    }

    ~YoloApp() {
        for (int i = 0; i < ofmaps.size(); i++) {
            delete[] ofmaps[i];  // Clean up memory
        }
        delete pipe_;
    }

    float get_avg_fps() const {
        float sum_fps = 0.0;
        for (const auto& fps : history_fps) {
            sum_fps += fps;
        }
        return sum_fps / history_fps.size();
    }
};

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);  // Set up signal handler
    vector<string> video_src_list;

    std::string video_str = "cam:0";

    // Iterate through the arguments
    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];

        // Handle -d or --dfp_path
        if (arg == "-d" || arg == "--dfp_path") {
            if (i + 1 < argc &&
                argv[i + 1][0] !=
                        '-') {  // Ensure there's a next argument and it is not another option
                model_path = argv[++i];
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle --video_paths
        else if (arg == "--video_paths") {
            if (i + 1 < argc &&
                argv[i + 1][0] !=
                        '-') {  // Ensure there's a next argument and it is not another option
                video_str = argv[++i];
                size_t pos = 0;
                std::string token;
                std::string delimiter = ",";
                while ((pos = video_str.find(delimiter)) != std::string::npos) {
                    token = video_str.substr(0, pos);
                    video_src_list.push_back(token);
                    video_str.erase(0, pos + delimiter.length());
                }
                video_src_list.push_back(video_str);
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }

        else if (arg == "--show") {
            // Creating GuiView for display
            gui = new MxQt(argc, argv);
        }
        // Handle unknown options
        else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // if video_paths arg isn't passed - use default video string.
    if (video_src_list.size() == 0) {
        video_src_list.push_back(video_str);
    }

    // Create the Accl object and load the DFP model
    std::vector<int> device_ids = {0};
    std::array<bool, 2> use_model_shape = {false, false};
    MX::Runtime::MxAccl accl{model_path, device_ids, use_model_shape};

    // Connect the post-processing model
    // accl.connect_post_model(fs::path(postprocessing_model_path));

    // Creating GuiView for display
    if (gui) {
        gui->screens[0]->SetSquareLayout(video_src_list.size(), false);  // Single stream layout
    }

    // Creating YoloApp objects for each video stream
    std::vector<YoloApp*> yolo_objs;
    for (int i = 0; i < video_src_list.size(); ++i) {
        YoloApp* obj = new YoloApp(&accl, video_src_list[i], i);
        yolo_objs.push_back(obj);
    }

    // Run the accelerator and wait
    accl.start();
    if (gui)
        gui->Run();  // Wait until the exit button is pressed in the Qt window
    else
        accl.wait();
    accl.stop();

    // print average FPS for each stream
    for (int i = 0; i < yolo_objs.size(); ++i) {
        std::cout << "Stream " << i << " Average FPS: " << yolo_objs[i]->get_avg_fps() << "\n";
    }

    // Cleanup
    delete gui;
    for (int i = 0; i < video_src_list.size(); ++i) {
        delete yolo_objs[i];
    }
}
