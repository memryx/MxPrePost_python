"""
============
Information:
============
Project: YOLOv8 example code on MXA
File Name: app.py

============
Description:
============
A script to show how to use the MultiStreamAcclerator API to perform a real-time inference
on MX3 using YOLOv8 model.
"""

###################################################################################################

# Imports
import time
import argparse
import numpy as np
import cv2
from queue import Queue, Full
import queue
from threading import Thread
from memryx import MultiStreamAsyncAccl
# from yolov8 import YoloV8 as YoloModel
from memryx_posts import Post
from collections import defaultdict
import sys

video_stream_1 = '/home/mixtile/memryx/media/people_1.mp4'
video_stream_2 = '/home/mixtile/memryx/media/dataset_2.mp4'
#video_stream_2 = 'rtsp://192.168.1.191:8554/video2'
dfp = '/home/mixtile/memryx/weights/YOLO_v8_small_640_640_3_tflite.dfp'
post_onnx_path = '/home/mixtile/memryx/weights/YOLO_v8_small_640_640_3_tflite_post.tflite'


FPS_LOG_INTERVAL = 30  # print out FPS every X frames

###################################################################################################

class Yolo8sMxa:
    """
    A demo app to run YOLOv8 on the MemryX MXA.
    """

###################################################################################################
    def __init__(self, video_paths, model_type, show=True):
        """
        Initialization function.
        """
        # Display control and stream initialization
        self.show = show
        self.done = False
        self.num_streams = len(video_paths)  # Number of streams

        # Stream-related containers and initialization
        self.streams = []
        self.streams_idx = [True] * self.num_streams
        self.stream_window = [False] * self.num_streams
        self.cap_queue = {i: Queue(maxsize=50) for i in range(self.num_streams)}
        self.dets_queue = {i: Queue(maxsize=50) for i in range(self.num_streams)}
        self.outputs = {i: [] for i in range(self.num_streams)}
        self.dims = {}
        self.color_wheel = {}
        self.model = {}
        self.model_type = model_type

        # Timing and FPS related
        self.dt_index = {i: 0 for i in range(self.num_streams)}
        self.frame_end_time = {i: 0 for i in range(self.num_streams)}
        self.fps = {i: 0 for i in range(self.num_streams)}
        self.dt_array = {i: np.zeros(30) for i in range(self.num_streams)}
        self.writer = {i: None for i in range(self.num_streams)}
        self.srcs_are_cams = {i: True for i in range(self.num_streams)}
        self.frame_count = {i: 0 for i in range(self.num_streams)}

        # FPS calculation related
        self.frame_count = defaultdict(int)
        self.start_ms = defaultdict(int)
        self.fps_number = defaultdict(float)
        self.history_fps = defaultdict(list)

        # Initialize video captures, models, and dimensions for each stream
        for i, video_path in enumerate(video_paths):
            if "/dev/video" in video_path:
                self.srcs_are_cams[i] = True
            else:
                self.srcs_are_cams[i] = False

            vidcap = cv2.VideoCapture(video_path)
            self.streams.append(vidcap)

            # Get frame dimensions
            self.dims[i] = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.color_wheel[i] = np.random.randint(0, 255, (20, 3)).astype(np.int32)

            # Initialize the YOLOv8 model
            # self.model[i] = YoloModel(stream_img_size=(self.dims[i][1], self.dims[i][0], 3), model_type=self.model_type)
            self.post = Post(model_type="numpy")


        # Start display thread
        if self.show:
            self.display_thread = Thread(target=self.display)

###################################################################################################
    def run(self):
        """
        Start inference on the MXA using multiple streams.
        """
        if self.model_type != "numpy":
            accl = MultiStreamAsyncAccl(
                dfp=self.dfp
            )  # Initialize the accelerator with DFP
            accl.set_postprocessing_model(
                self.post_model, model_idx=0
            )  # Set the post-processing model
        else:
            accl = MultiStreamAsyncAccl(
                dfp=self.dfp, use_model_shape=(False, False)
            )  # Initialize the accelerator with DFP

        if self.show:
            self.display_thread.start()  # Start the display thread

        # Connect input and output streams for the accelerator
        accl.connect_streams(self.capture_and_preprocess, self.postprocess, self.num_streams)
        accl.wait()

        self.done = True

        # Join display thread
        if self.show:
            self.display_thread.join()

###################################################################################################
    def capture_and_preprocess(self, stream_idx):
        """
        Captures a frame for the video device and pre-processes it.
        """
        # if self.srcs_are_cams[stream_idx]:
        while True:
            got_frame, frame = self.streams[stream_idx].read()

            if not got_frame or self.done:
                self.streams_idx[stream_idx] = False
                return None

            if self.srcs_are_cams[stream_idx] and self.cap_queue[stream_idx].full():
                # drop frame
                continue
            else:
                if self.show:
                    # Put the frame in the cap_queue to be processed later
                    try:
                        self.cap_queue[stream_idx].put(frame, timeout=2)
                    except Full:
                        print('Dropped frame')
                        continue
                    
                # Pre-process the frame using the corresponding model
                # frame = self.model[stream_idx].preprocess(frame)
                frame = self.post.preprocess(frame)
                return frame


###################################################################################################
    def postprocess(self, stream_idx, *mxa_output):
        """
        Post-process the output from MXA.
        """
        # dets = self.model[stream_idx].postprocess(mxa_output)  # Get detection results
        dets = self.post.postprocess(mxa_output)  # Get detection results

        # Queue detection results for display
        if self.show:
            self.dets_queue[stream_idx].put(dets)

        # Calculate FPS
        self.update_fps(stream_idx)
        
    def update_fps(self, stream_idx):

        # increment frame count
        self.frame_count[stream_idx] += 1
        
        now_ms = int(time.time() * 1000)

        if self.frame_count[stream_idx] == 1:
            # record start time
            self.start_ms[stream_idx] = now_ms
        else:
            # print FPS
            if self.frame_count[stream_idx] % FPS_LOG_INTERVAL == 0:
                
                # msg
                msg = "Frame cnt: {} stream {} => FPS: {:.2f}"
                lines = [msg.format(self.frame_count[i], i, self.fps_number[i]) for i in range(self.num_streams)]
                print("\n".join(lines))

                # Overwrite previous msg
                sys.stdout.write(f"\033[{self.num_streams}A")
                sys.stdout.flush()

                # Update history
                for i in range(self.num_streams):
                    self.history_fps[i].append(self.fps_number[i])

            # update fps_number
            duration_ms = now_ms - self.start_ms[stream_idx]
            self.fps_number[stream_idx] = (self.frame_count[stream_idx] * 1000.0) / duration_ms

    def get_avg_fps(self, stream_idx):
        if self.history_fps[stream_idx]:
            return np.mean(self.history_fps[stream_idx])
        return 0

###################################################################################################
    def display(self):
        """
        Displays the processed frames with detections in separate windows.
        """
        while not self.done:
            # Iterate through each stream for displaying frames
            for stream_idx in range(self.num_streams):
                
                try:
                    # Python blocky queue, no need to check if not queue.empty()
                    frame = self.cap_queue[stream_idx].get(timeout=2)
                    dets = self.dets_queue[stream_idx].get(timeout=2)
                except queue.Empty:
                    break 

                self.cap_queue[stream_idx].task_done()
                self.dets_queue[stream_idx].task_done()

                # Draw detection boxes
                for d in dets:
                    x1, y1, w, h = d.xywh
                    color = tuple(int(c) for c in self.color_wheel[stream_idx][d.class_id % 20])

                    # Draw bounding boxes
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                    # Add class label
                    frame = cv2.putText(frame, d.class_name, (x1 + 2, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Add FPS to frame
                fps_text = f"{self.fps_number[stream_idx]:.2f}"
                frame = cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                window_name = f"Stream {stream_idx} - YOLOv8"
                cv2.imshow(window_name, frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                self.done = True

        # Close all windows and release resources after processing
        cv2.destroyAllWindows()
        for stream in self.streams:
            stream.release()

###################################################################################################

def main(args):
    """
    Main function to start YOLOv8 inference.
    """
    if args.post_model.endswith('.onnx'):
        model_type = 'onnx'
    elif args.post_model.endswith('.tflite'):
        model_type = 'tflite'
    elif args.post_model == "numpy":
        model_type = 'numpy'
    else:
        raise ValueError(f"Unsupported post-processing model format: {args.post_model}")

    # Initialize the application with video paths and display settings
    yolo8s_inf = Yolo8sMxa(video_paths=args.video_paths, model_type=model_type, show=args.show)
    yolo8s_inf.dfp = args.dfp  # Set the DFP path from arguments
    yolo8s_inf.post_model = args.post_model  # Set the post-processing model path from arguments
    yolo8s_inf.run()  # Start inference

    # Print final average FPS for each stream
    for i in range(yolo8s_inf.num_streams):
        sys.stdout.write("\033[F\033[K")
    for i in range(yolo8s_inf.num_streams):
        print(f'Final Avg FPS for Stream {i}: {yolo8s_inf.get_avg_fps(i):.2f}')


###################################################################################################

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="\033[34mMemryX YoloV8 Demo\033[0m")
    
    # Video input paths
    parser.add_argument('--video_paths', nargs='+', dest="video_paths", 
                        action="store", 
                        default=[video_stream_2],
                        help="Path to video files for inference. Use '/dev/video0' for webcam. (Default:'/dev/video0')")
    
    # Option to turn on display
    parser.add_argument('--show', action ='store_true', help="Display cartoonized video")

    # DFP model argument
    parser.add_argument('-d', '--dfp', type=str, 
                        default=dfp, 
                        help="Path to the compiled DFP file (default: 'models/tflite/YOLO_v8_small_640_640_3_tflite.dfp')")

    # Post-processing model argument
    parser.add_argument('-p', '--post_model', type=str, 
                        default=post_onnx_path, 
                        help="Path to the post-processing ONNX file (default: 'models/tflite/YOLO_v8_small_640_640_3_tflite_post.tflite')")

    args = parser.parse_args()

    # Call the main function
    main(args)

# eof
