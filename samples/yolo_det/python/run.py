# Imports
import time
import argparse
import numpy as np
import cv2
from queue import Queue
import queue
from threading import Thread
from collections import defaultdict
import sys
import mxpipe

video_stream_1 = "/home/mixtile/memryx/media/people_1.mp4"
video_stream_2 = "/home/mixtile/memryx/media/dataset_2.mp4"
dfp = "/home/mixtile/memryx/weights/YOLO_v8_small_640_640_3_tflite.dfp"
post_onnx_path = (
    "/home/mixtile/memryx/weights/YOLO_v8_small_640_640_3_tflite_post.tflite"
)


FPS_LOG_INTERVAL = 30  # print out FPS every X frames


class YoloApp:
    """
    A demo app to run YOLO on the MemryX MXA.
    """

    def __init__(self, args):
        """
        Initialization function.
        """

        self.show = args.show
        self.old_bind = args.old_bind

        # Display control and stream initialization
        self.done = False
        self.num_streams = len(args.video_paths)  # Number of streams

        # Stream-related containers and initialization
        self.streams = []
        self.cap_queue = {i: Queue(maxsize=50) for i in range(self.num_streams)}
        self.result_queue = {i: Queue(maxsize=50) for i in range(self.num_streams)}
        self.srcs_are_cams = {i: True for i in range(self.num_streams)}

        # FPS calculation related
        self.frame_count = defaultdict(int)
        self.start_ms = defaultdict(int)
        self.fps_number = defaultdict(float)
        self.history_fps = defaultdict(list)

        # Initialize video captures, models, and dimensions for each stream
        for i, video_path in enumerate(args.video_paths):
            if "/dev/video" in video_path:
                self.srcs_are_cams[i] = True
            else:
                self.srcs_are_cams[i] = False

            vidcap = cv2.VideoCapture(video_path)
            self.streams.append(vidcap)

        # init MXPipe pipeline
        self.pipe = mxpipe.Pipeline(
            task="yolov8_det",
            ori_width=int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            ori_height=int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            conf=0.3,
            iou=0.4,
            valid_classes=[0]
        )

        # Start display thread
        if self.show:
            self.display_thread = Thread(target=self.display)

    def run(self):
        """
        Start inference on the MXA using multiple streams.
        """
        if self.show:
            self.display_thread.start()  # Start the display thread

        if self.old_bind:
            print("Run with old binding")
            import memryx

            accl = memryx.MultiStreamAsyncAccl(
                dfp=self.dfp, use_model_shape=(False, False)
            )
            accl.connect_streams(
                self.in_callback_old_bind, self.out_callback_old_bind, self.num_streams
            )
            accl.wait()

        else:
            print("Run with new binding")
            import mxapi

            local = False
            accl = mxapi.MxAccl(self.dfp, [0], [False, False], local)
            # accl.connect_post_model(args.post_model)

            for i in range(self.num_streams):
                accl.connect_stream(self.in_callback, self.out_callback, stream_id=i)

            accl.start()
            accl.wait()

        self.done = True

        # Join display thread
        if self.show:
            self.display_thread.join()

    def in_callback(self, stream_id):
        """
        Captures a frame for the video device and pre-processes it.
        """
        # if self.srcs_are_cams[stream_id]:
        while True:
            got_frame, frame = self.streams[stream_id].read()

            if not got_frame or self.done:
                return None

            if self.srcs_are_cams[stream_id] and self.cap_queue[stream_id].full():
                # drop frame
                continue

            if self.show:
                # Put the frame in the cap_queue to be processed later
                self.cap_queue[stream_id].put(frame)

            # TODO: get frame with shape based on use_model_shape
            frame = self.pipe.preprocess(frame)
            frame = frame.reshape(640, 640, 1, 3)

            return frame

    def out_callback(self, mxa_output, stream_id):
        """
        Post-process the output from MXA.
        """

        # call postprocess from mxpipe
        result = self.pipe.postprocess(mxa_output)

        # Queue detection results for display
        if self.show:
            self.result_queue[stream_id].put(result)

        # Calculate FPS
        self.update_fps(stream_id)

    # for old binding test
    def in_callback_old_bind(self, stream_id):
        return self.in_callback(stream_id)

    # for old binding test
    def out_callback_old_bind(self, stream_id, *ofmaps):
        self.out_callback(list(ofmaps), stream_id)

    def display(self):
        """
        Displays the processed frames with detections in separate windows.
        """
        while not self.done:
            # Iterate through each stream for displaying frames
            for stream_id in range(self.num_streams):
                try:
                    # Python blocky queue, no need to check if not queue.empty()
                    frame = self.cap_queue[stream_id].get(timeout=2)
                    result = self.result_queue[stream_id].get(timeout=2)
                except queue.Empty:
                    break

                # Draw detections on the frame
                display_img = self.pipe.draw(frame, result)

                # Add FPS to frame
                fps_text = f"FPS: {self.fps_number[stream_id]:.2f}"
                display_img = cv2.putText(
                    display_img,
                    fps_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                # show
                window_name = f"Stream {stream_id} - YOLO Detection"
                cv2.imshow(window_name, display_img)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) == ord("q"):
                self.done = True

        # Close all windows and release resources after processing
        cv2.destroyAllWindows()
        for stream in self.streams:
            stream.release()

    def update_fps(self, stream_id):
        # increment frame count
        self.frame_count[stream_id] += 1

        now_ms = int(time.time() * 1000)

        if self.frame_count[stream_id] == 1:
            # record start time
            self.start_ms[stream_id] = now_ms
        else:
            # print FPS
            if self.frame_count[stream_id] % FPS_LOG_INTERVAL == 0:
                # msg
                msg = "Frame cnt: {} stream {} => FPS: {:.2f}"
                lines = [
                    msg.format(self.frame_count[i], i, self.fps_number[i])
                    for i in range(self.num_streams)
                ]
                print("\n".join(lines))

                # Overwrite previous msg
                sys.stdout.write(f"\033[{self.num_streams}A")
                sys.stdout.flush()

                # Update history
                for i in range(self.num_streams):
                    self.history_fps[i].append(self.fps_number[i])

            # update fps_number
            duration_ms = now_ms - self.start_ms[stream_id]
            self.fps_number[stream_id] = (
                self.frame_count[stream_id] * 1000.0
            ) / duration_ms

    def get_avg_fps(self, stream_id):
        if self.history_fps[stream_id]:
            return np.mean(self.history_fps[stream_id])
        return 0


def main(args):
    """
    Main function to start YOLO inference.
    """

    # Initialize the application with video paths and display settings
    app = YoloApp(args)
    app.dfp = args.dfp  # Set the DFP path from arguments
    app.run()  # Start inference

    for i in range(app.num_streams):
        print(f"Final Avg FPS for Stream {i}: {app.get_avg_fps(i):.2f}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="\033[34mMemryX YoloV8 Demo\033[0m")

    # Video input paths
    parser.add_argument(
        "--video_paths",
        nargs="+",
        dest="video_paths",
        action="store",
        default=[video_stream_2],
        help="Path to video files for inference. Use '/dev/video0' for webcam. (Default:'/dev/video0')",
    )

    # Option to turn on display
    parser.add_argument("--show", action="store_true", help="Display results")

    # DFP model argument
    parser.add_argument(
        "-d",
        "--dfp",
        type=str,
        default=dfp,
        help="Path to the compiled DFP file (default: 'models/tflite/YOLO_v8_small_640_640_3_tflite.dfp')",
    )

    parser.add_argument(
        "--old_bind", action="store_true", help="Use old binding method"
    )

    args = parser.parse_args()

    # Call the main function
    main(args)
