import time
import argparse
import numpy as np
import cv2
from queue import Queue
import queue
from threading import Thread
from collections import defaultdict
import sys
import mxprepost
from memryx import mxapi

FPS_LOG_INTERVAL = 30  # print out FPS every X frames


class YoloApp:
    """
    A demo app to run YOLO on the MemryX MXA.
    """

    def __init__(self, args):
        self.show = not args.no_show

        # Display control and stream initialization
        self.done = False
        self.num_streams = len(args.video_paths)

        # Stream-related containers and initialization
        self.streams = []
        self.cap_queue = {i: Queue(maxsize=50) for i in range(self.num_streams)}
        self.result_queue = {i: Queue(maxsize=50) for i in range(self.num_streams)}
        self.srcs_are_cams = {i: True for i in range(self.num_streams)}

        # Per-stream original dimensions
        self.ori_width = {}
        self.ori_height = {}

        # FPS calculation related
        self.frame_count = defaultdict(int)
        self.start_ms = defaultdict(int)
        self.fps_number = defaultdict(float)
        self.history_fps = defaultdict(list)

        # Init captures
        for i, video_path in enumerate(args.video_paths):
            self.srcs_are_cams[i] = ("/dev/video" in video_path)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {video_path}")

            self.streams.append(cap)

            # Store each stream's dimensions
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.ori_width[i] = w
            self.ori_height[i] = h

        if self.show:
            self.display_thread = Thread(target=self.display, daemon=True)

        # store args for run()
        self.args = args

    def run(self):
        """
        Start inference on the MXA using multiple streams.
        """
        if self.show:
            self.display_thread.start()

        local = False
        accl = mxapi.MxAccl(self.dfp, [0], [False, False], local)

        for i in range(self.num_streams):
            accl.connect_stream(self.in_callback, self.out_callback, stream_id=i)

        # init mxprepost pipeline (shared across streams)
        self.prepost = mxprepost.MxPrepost(
            accl=accl,
            task=self.args.task,
            conf=0.3,
            iou=0.4,
            # classmap_path="classes.txt",
            # valid_classes=[0],
            # model_id=0
        )

        accl.start()
        accl.wait()

        self.done = True

        if self.show:
            self.display_thread.join()

    def in_callback(self, stream_id):
        """
        Capture + preprocess for a given stream.
        """
        while True:
            got_frame, frame = self.streams[stream_id].read()

            if not got_frame or self.done:
                return None

            # Drop frames if camera + display queue is full
            if self.srcs_are_cams[stream_id] and self.show and self.cap_queue[stream_id].full():
                continue

            if self.show:
                self.cap_queue[stream_id].put(frame)

            # call preprocess from mxprepost
            frame = self.prepost.preprocess(frame)

            return frame

    def out_callback(self, mxa_output, stream_id):
        """
        Postprocess per stream using that stream's original dimensions.
        """

        result = self.prepost.postprocess(mxa_output, self.ori_width[stream_id], self.ori_height[stream_id])
        if self.show:
            # Drop results if display queue is full (avoid deadlock)
            if self.result_queue[stream_id].full():
                try:
                    _ = self.result_queue[stream_id].get_nowait()
                except queue.Empty:
                    pass
            self.result_queue[stream_id].put(result)

        # Calculate FPS
        self.update_fps(stream_id)

    def display(self):
        while not self.done:
            for stream_id in range(self.num_streams):
                try:
                    frame = self.cap_queue[stream_id].get(timeout=2)
                    result = self.result_queue[stream_id].get(timeout=2)
                except queue.Empty:
                    continue

                # Draw detections on the frame
                display_img = self.prepost.draw(frame, result)

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

        cv2.destroyAllWindows()
        for cap in self.streams:
            cap.release()

    def update_fps(self, stream_id):
        self.frame_count[stream_id] += 1
        now_ms = int(time.time() * 1000)

        if self.frame_count[stream_id] == 1:
            # record start time
            self.start_ms[stream_id] = now_ms
            return

        if self.frame_count[stream_id] % FPS_LOG_INTERVAL == 0:
            msg = "Frame cnt: {}, Stream {} => FPS: {:.2f}"
            lines = [
                msg.format(self.frame_count[i], i, self.fps_number[i])
                for i in range(self.num_streams)
            ]
            print("\n".join(lines))

            # Overwrite previous msg
            sys.stdout.write(f"\033[{self.num_streams}A")
            sys.stdout.flush()

            for i in range(self.num_streams):
                self.history_fps[i].append(self.fps_number[i])

            # update fps_number
            duration_ms = now_ms - self.start_ms[stream_id]
            self.fps_number[stream_id] = (
                self.frame_count[stream_id] * 1000.0
            ) / duration_ms

    def get_avg_fps(self, stream_id):
        if self.history_fps[stream_id]:
            return float(np.mean(self.history_fps[stream_id]))
        return 0.0


def main(args):
    """
    Main function to start YOLO inference.
    """

    # Initialize the application with video paths and display settings
    app = YoloApp(args)
    app.dfp = args.dfp  # Set the DFP path from arguments
    app.run()  # Start inference

    for i in range(app.num_streams):
        print(f"\n\nFinal Avg FPS for Stream {i}: {app.get_avg_fps(i):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="\033[34mMemryX YOLO Demo\033[0m")

    parser.add_argument(
        "--video_paths",
        nargs="+",
        dest="video_paths",
        action="store",
        default=["/dev/video0"],
        help="Input sources. Use '/dev/video0' for webcam. (Default: '/dev/video0')",
    )

    parser.add_argument("--no-show", action="store_true", help="Disable display window")

    parser.add_argument(
        "-d",
        "--dfp",
        type=str,
        required=True,
        help="Path to the compiled DFP file",
    )

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        help="Task name (e.g. 'yolov8_det', 'yolov8_seg', 'yolov11_pose')",
    )

    args = parser.parse_args()
    main(args)
