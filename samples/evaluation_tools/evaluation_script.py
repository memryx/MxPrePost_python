import time
import cv2
import numpy as np
import mxprepost
import mxapi


def run_async(
    dfp_path: str,
    source: str,
    task: str = None,
    post_model: str = None,
    max_frames: int = 500,
) -> None:
    # Initialize accelerator
    local = False
    # Post-processing model requires use_model_shape[1] = True
    use_model_shape = [False, True] if post_model else [False, False]
    accl = mxapi.MxAccl(dfp_path, [0], use_model_shape, local)
    if post_model:
        accl.connect_post_model(post_model)    
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    disp_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    disp_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize mxprepost for preprocessing and postprocessing (only if not using post-processing model)
    prepost = None
    if not post_model:
        if task is None:
            raise ValueError("Task must be provided when not using a post-processing model")
        prepost = mxprepost.MxPrepost(
            accl=accl,
            task=task,
            ori_width=disp_w,
            ori_height=disp_h,
            conf=0.3,
            iou=0.4,
        )
    
    # Track timing data
    time_stamps = []
    timed_frames = 0

    def data_source_generator():
        """Generator function that yields preprocessed frames."""
        nonlocal timed_frames
        
        while True:
            if timed_frames >= max_frames:
                break
            
            ret, frame = cap.read()
            if not ret:
                # Video ended, restart from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break  # Can't read, stop generator
            
            # Preprocess frame
            if prepost:
                img = prepost.preprocess(frame)
            else:
                # Manual preprocessing when using post-processing model
                # Resize to 640x640 (YOLO standard input size)
                frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                # Convert to float32 and normalize to [0, 1]
                img = frame_resized.astype(np.float32) / 255.0
                # Reshape to [H, W, 1, C] format required when use_model_shape[0] = False
                img = img.reshape(640, 640, 1, 3)
            
            timed_frames += 1
            yield img
        
        cap.release()

    # Create the generator object
    data_source = data_source_generator()

    def postprocess_callback(mxa_output, stream_id):
        """Callback for processing model outputs."""
        nonlocal time_stamps
        
        # Use mxprepost for postprocessing (if available)
        if prepost:
            result = prepost.postprocess(mxa_output)
        else:
            _ = mxa_output  # Acknowledge outputs exist
        
        # Store timestamp in seconds
        time_stamps.append(time.perf_counter())

    accl.connect_stream(data_source, postprocess_callback, stream_id=0)
    accl.start()
    accl.wait()

    if len(time_stamps) >= 2:
        print("\n" + "="*60)
        print(f"Total frames: {timed_frames}")
        print("="*60)
        
        # Frame-to-Frame (Throughput): time between consecutive outputs
        time_deltas = [t2 - t1 for t2, t1 in zip(time_stamps[1:], time_stamps[:-1])]
        avg_delta = sum(time_deltas) / len(time_deltas)  # In seconds
        avg_delta_ms = avg_delta * 1000  # Convert to milliseconds
        print(f"Frame-to-Frame Latency: {avg_delta_ms:0.2f} ms")
        throughput_fps = 1.0 / avg_delta if avg_delta > 0 else 0
        print(f"Throughput FPS: {throughput_fps:0.2f}")
        print("="*60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FPS measurement tool")
    parser.add_argument("-d", "--dfp", type=str, required=True, help="Path to DFP file")
    parser.add_argument("-t", "--task", type=str, required=False, 
                       help="Task type (e.g. 'yolov8_det', 'yolov8_seg'). Required only when not using -p")
    parser.add_argument("-s", "--source", type=str, default="/dev/video0", help="Video source")
    parser.add_argument("-f", "--frames", type=int, default=1000, help="Max frames to process")
    parser.add_argument("-p", "--post_model", type=str, default=None, 
                       help="Path to the post-processing ONNX model (optional)")
    args = parser.parse_args()

    # Validate that either task or post_model is provided
    if not args.post_model and not args.task:
        parser.error("Either --task or --post_model must be provided")

    run_async(
        dfp_path=args.dfp,
        source=args.source,
        task=args.task,
        post_model=args.post_model,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    main()