#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>

namespace fs = std::filesystem;

std::atomic_bool runflag;
bool use_tflite = false;  // Flag to determine whether to use ONNX or TFLite

//YoloV7 application specific parameters
// fs::path model_path = "/home/jackyyeh/memryx/models/yolo/tflite/yolov8n.tflite/yolov8n_tflite.dfp";
// fs::path postprocessing_model_path = "/home/jackyyeh/memryx/models/yolo/tflite/yolov8n.tflite/yolov8n_post.tflite";
fs::path model_path = "models/yolo/YOLO_v8_nano_640_640_3_onnx.dfp";                      // Default model path
fs::path postprocessing_model_path = "models/yolo/YOLO_v8_nano_640_640_3_onnx_post.onnx"; // Default post-processing model path
// fs::path default_videoPath = "videos/sample.mp4";
#define AVG_FPS_CALC_FRAME_COUNT  50

//signal handler
void signal_handler(int p_signal)
{
    runflag.store(false);
}

struct Box {
    float x1, y1, x2, y2, confidence, class_id;
};

// In case of cameras try to use best possible input configurations which are setting the
// resolution to 640x480 and try to set the input FPS to 30
bool configureCamera(cv::VideoCapture &vcap)
{
    bool settings_success = true;

    try {
        if (!vcap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) ||
                !vcap.set(cv::CAP_PROP_FRAME_WIDTH, 640) ||
                !vcap.set(cv::CAP_PROP_FPS, 30)) {
            std::cout << "Setting vcap Failed\n";
            cv::Mat simpleframe;
            if (!vcap.read(simpleframe)) {
                settings_success = false;
            }
        }
    }
    catch (...) {
        std::cout << "Exception occurred while setting properties\n";
        settings_success = false;
    }

    return settings_success;
}

// Tries to open the camera with custom settings set in configureCamera
// If not possible, open it with default settings
bool openCamera(cv::VideoCapture &vcap, int device, int api)
{
    vcap.open(device, api);
    if (!vcap.isOpened()) {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configureCamera(vcap)) {
        vcap.release();
        vcap.open(device, api);
        if (vcap.isOpened()) {
            std::cout << "Reopened vcap with original resolution\n";
        }
        else {
            std::cerr << "Failed to reopen vcap\n";
            return false;
        }
    }

    return true;
}

class YoloV8
{
  private:
    // Model Params
    int model_input_width;  // width of model input image
    int model_input_height; // height of model input image
    int input_image_width;  // width of input image
    int input_image_height; // height of input image
    int num_boxes = 8400;   // YOLOv8 has 8400 anchor points
    int features_per_box = 84; // number of output features
    float conf_thresh = 0.6; // Confidence threshold of the boxes
    float nms_thresh = 0.5;  // IoU threshold for non-maximum suppression
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // Application Variables
    std::deque<cv::Mat> frames_queue;
    std::mutex frame_queue_mutex;
    int num_frames = 0;
    int frame_count = 0;
    float fps_number = .0;
    std::chrono::milliseconds start_ms;
    cv::VideoCapture vcap;
    std::vector<size_t> in_tensor_sizes;
    std::vector<size_t> out_tensor_sizes;
    MX::Types::MxModelInfo model_info;
    float* mxa_output;
    cv::Mat displayImage;
    MxQt* gui_;
    int length;
    std::vector<Box> all_boxes;
    std::vector<float> all_scores;
    std::vector<cv::Rect> cv_boxes;

    // Function to preprocess the input image (resize and normalize)
    cv::Mat preprocess(cv::Mat &image)
    {
        // Get original image dimensions
        int img_height = image.rows;
        int img_width = image.cols;

        // Determine the size of the square image (longest side)
        length = std::max(img_height, img_width);

        // Create a black square image with the same number of channels
        cv::Mat squareImage = cv::Mat::zeros(cv::Size(length, length), image.type());

        // Copy the original image into the square image
        image.copyTo(squareImage(cv::Rect(0, 0, img_width, img_height)));

        // Resize to 640x640 for the model input
        cv::Mat resizedImage;
        cv::resize(squareImage, resizedImage, cv::Size(640, 640), cv::INTER_LINEAR);

        // Convert to float32 and normalize pixel values (0-1 range)
        resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
        
        // Convert HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        // std::vector<cv::Mat> channels(3);
        // cv::split(resizedImage, channels);
        // cv::Mat chwImage;
        // cv::merge(channels, chwImage);

        return resizedImage;
    }

    std::vector<Box> get_detections(float* ofmap, int num_boxes, const cv::Mat &inframe)
    {
        std::vector<Box> all_boxes;
        std::vector<cv::Rect> cv_boxes;
        std::vector<float> all_scores;
        std::vector<Box> filtered_boxes;

        // Precompute scaling factors once
        const float y_factor = static_cast<float>(length) / static_cast<float>(model_input_height);
        const float x_factor = static_cast<float>(length) / static_cast<float>(model_input_width);

        // Iterate through the detections
        for (int i = 0; i < num_boxes; ++i) {
            // Extract coordinates and size for each box (pre-calculated)
            float x0 = ofmap[i];                   // x center
            float y0 = ofmap[num_boxes + i];       // y center
            float w = ofmap[2 * num_boxes + i];    // width
            float h = ofmap[3 * num_boxes + i];    // height

            // Scale the box coordinates to the original image size
            x0 *= x_factor;
            y0 *= y_factor;
            w *= x_factor;
            h *= y_factor;

            // Compute the top-left and bottom-right coordinates of the box
            int x1 = static_cast<int>(x0 - 0.5f * w);
            int y1 = static_cast<int>(y0 - 0.5f * h);
            int x2 = static_cast<int>(x0 + 0.5f * w);
            int y2 = static_cast<int>(y0 + 0.5f * h);

            // Iterate through the classes
            for (int j = 4; j < features_per_box; ++j) {
                float confidence = ofmap[j * num_boxes + i];

                if (confidence > conf_thresh) {
                    // Add the box to the list if confidence is greater than threshold
                    Box box;
                    box.x1 = x1;
                    box.y1 = y1;
                    box.x2 = x2;
                    box.y2 = y2;
                    box.class_id = j - 4;  // Adjust class id to start from 0
                    box.confidence = confidence;

                    all_boxes.push_back(box);
                    all_scores.push_back(confidence);
                    cv_boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                }
            }
        }

        // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
        std::vector<int> nms_result;
        if (!cv_boxes.empty()) {
            cv::dnn::NMSBoxes(cv_boxes, all_scores, conf_thresh, nms_thresh, nms_result);

            // Filter detections based on NMS result
            for (int idx : nms_result) {
                filtered_boxes.push_back(all_boxes[idx]);
            }
        }

        return filtered_boxes;
    }

    // Function to draw bounding boxes
    void draw_bounding_boxes(cv::Mat &image, const std::vector<Box> &boxes)
    {
        for (const Box &box : boxes) {
            // Draw rectangle
            cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);

            // Display confidence score as a label
            std::string label = class_names[box.class_id];
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(image, label, cv::Point(box.x1, box.y1 - labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                        1);
        }
    }


    bool incallback_getframe(vector<const MX::Types::FeatureMap*> dst, int streamLabel)
    {
        if (runflag.load()) {
            cv::Mat inframe;
            cv::Mat rgbImage;
            bool got_frame = vcap.read(inframe);

            if (!got_frame) {
                std::cout << "No frame \n\n\n";
                vcap.release();
                return false;
            }

            cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
            {
                std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                frames_queue.push_back(rgbImage);
            }

            cv::Mat preProcframe = preprocess(rgbImage);
            if(!use_tflite) {
                cv::Mat chw_image;
                cv::dnn::blobFromImage(preProcframe, chw_image, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0,0,0), false, false);
                preProcframe = chw_image;
            }

            dst[0]->set_data((float*)preProcframe.data);

            return true;
        }
        else {
            vcap.release();
            return false;
        }
    }

    bool outcallback_getmxaoutput(vector<const MX::Types::FeatureMap*> src, int streamLabel)
    {
        src[0]->get_data(mxa_output);

        {
            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
            displayImage = frames_queue.front();
            frames_queue.pop_front();
        }

        std::vector<Box> detected_objectVector = get_detections(mxa_output, num_boxes, displayImage);
        // draw_bounding_boxes(displayImage, detected_objectVector);

        // gui_->screens[0]->SetDisplayFrame(streamLabel, &displayImage, fps_number);

        frame_count++;
        if (frame_count == 1) {
            start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        }
        else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0) {
            std::chrono::milliseconds duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
            fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());
            std::cout << "Average FPS: " << fps_number << "\n";
            frame_count = 0;
        }
        return true;
    }

  public:
    YoloV8(MX::Runtime::MxAccl* accl, std::string video_src, MxQt* gui, int index)
    {
        // Assigning gui variable to class-specific variable
        gui_ = gui;

        // If the input is a camera, try to use optimal settings
        if (video_src.substr(0, 3) == "cam") {
#ifdef __linux__
            if (!openCamera(vcap, 0, cv::CAP_V4L2)) {
                throw(std::runtime_error("Failed to open: " + video_src));
            }
#elif defined(_WIN32)
            if (!openCamera(vcap, 0, cv::CAP_ANY)) {
                throw(std::runtime_error("Failed to open: " + video_src));
            }
#endif
        }
        else if (video_src.substr(0, 3) == "vid") {
            vcap.open(video_src.substr(4), cv::CAP_ANY);
        }
        else {
            throw(std::runtime_error("Given video src: " + video_src + " is invalid" +
                                     "\n\n\tUse ./objectDetection cam:<camera index>,vid:<path to video file>\n\n"));
        }

        if (!vcap.isOpened()) {
            std::cout << "videocapture NOT opened \n";
            runflag.store(false);
        }

        // Getting input image dimensions
        input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
        input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // Retrieve model info of 0th model (assuming YOLOv8)
        model_info = accl->get_model_info(0);

        // Allocate memory for YOLOv8 output (84 parameters per anchor, with 8400 anchors)
        mxa_output = new float[num_boxes * features_per_box]; // YOLOv8 has 84 outputs per box

        // Getting model input shapes and display size
        model_input_height = model_info.in_featuremap_shapes[0][0];
        model_input_width = model_info.in_featuremap_shapes[0][1];

        // Bind the callback functions for input and output processing
        auto in_cb = std::bind(&YoloV8::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
        auto out_cb = std::bind(&YoloV8::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);

        // Connect the stream to the accelerator object
        accl->connect_stream(in_cb, out_cb, index /**Unique Stream Idx */, 0 /**Model Idx */);

        // Start the callbacks when the process begins
        runflag.store(true);
    }

    ~YoloV8()
    {
        delete[] mxa_output;
        mxa_output = nullptr;
    }
};


int main(int argc, char* argv[])
{
    signal(SIGINT, signal_handler);
    vector<string> video_src_list;

    //Create the Accl object and load the DFP
    MX::Runtime::MxAccl* accl;
    accl = new MX::Runtime::MxAccl(model_path);

    accl->connect_post_model(postprocessing_model_path);

    // Check if the post-processing model is a TFLite model
    use_tflite = postprocessing_model_path.extension() == ".tflite";

    for (int i = 1; i < argc; i++)
    {

        std::string arg = argv[i];

        // Handle --video_paths
        if (arg == "--video")
        {
            if (i + 1 < argc && argv[i + 1][0] != '-')
            { // Ensure there's a next argument and it is not another option
                std::string video_str = std::filesystem::path(argv[++i]);
                video_src_list.push_back("vid:" + video_str);
            }
            else
            {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                return 1;
            }
        }
        // Handle unknown options
        else
        {
            std::cerr << "Error: Unknown option " << arg << "\n";
            return 1;
        }
    }
    // if(argc <= 1) { //Default mode: runs an exisitng video file
    //     video_src_list.push_back("vid:" + default_videoPath.string());
    // }
    // else {   //Decoding comma seperated video input list
    //     std::string video_str(argv[1]);
    //     size_t pos = 0;
    //     std::string token;
    //     std::string delimiter = ",";
    //     while ((pos = video_str.find(delimiter)) != std::string::npos) {
    //         token = video_str.substr(0, pos);
    //         video_src_list.push_back(token);
    //         video_str.erase(0, pos + delimiter.length());
    //     }
    //     video_src_list.push_back(video_str);
    // }

    // Creating GuiView which is a memryx qt util for easy display
    // MxQt gui(argc, argv);
    // // Setting the layout of the display based on number of input streams. Full screen mode only when more than one stream
    // if(video_src_list.size() == 1) {
    //     gui.screens[0]->SetSquareLayout(1, false);
    // }
    // else {
    //     gui.screens[0]->SetSquareLayout(static_cast<int>(video_src_list.size()));
    // }

    //Creating a YoloV7 object for each stream which also connects the corresponding stream to accl.
    std::vector<YoloV8*>yolo_objs;
    for(int i = 0; i < video_src_list.size(); ++i) {
        // YoloV8* obj = new YoloV8(accl, video_src_list[i], &gui, i);
        YoloV8* obj = new YoloV8(accl, video_src_list[i], nullptr, i);
        yolo_objs.push_back(obj);
    }

    //Run the accelerator and wait
    accl->start();
    accl->wait();
    // gui.Run(); //This command waits for exit to be pressed in Qt window
    accl->stop();

    //Cleanup
    delete accl;
    for(int i = 0; i < video_src_list.size(); ++i ) {
        delete yolo_objs[i];
    }
}
