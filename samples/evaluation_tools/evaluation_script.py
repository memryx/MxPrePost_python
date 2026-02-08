import time
import numpy as np
import mxprepost
from memryx import mxapi


def run_async(
    dfp_path: str,
    task: str = None,
    post_model: str = None,
    max_frames: int = 500,
) -> None:
    # Initialize accelerator
    local = False

    use_model_shape = [False, True] if post_model else [False, False]

    accl = mxapi.MxAccl(dfp_path, [0], use_model_shape, local)
    if post_model:
        accl.connect_post_model(post_model)

    disp_w = 640
    disp_h = 640

    # Initialize mxprepost (only if not using post-processing model)
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

    base_frame = np.random.rand(640, 640, 1, 3).astype(np.float32)

    def data_source_generator():
        """Generator function that yields constant frames."""
        nonlocal timed_frames

        while timed_frames < max_frames:

            timed_frames += 1
            yield base_frame

    def postprocess_mxprepost_callback(mxa_output, stream_id):
        """Callback for processing model outputs."""
        nonlocal time_stamps

        _ = prepost.postprocess(mxa_output)

        time_stamps.append(time.perf_counter())

    def postprocess_callback(mxa_output, stream_id):
        """Callback for processing model outputs."""
        nonlocal time_stamps

        time_stamps.append(time.perf_counter())

    data_source = data_source_generator()
    
    if prepost:
        accl.connect_stream(data_source, postprocess_mxprepost_callback, stream_id=0)
    else:
        accl.connect_stream(data_source, postprocess_callback, stream_id=0)

    accl.start()
    accl.wait()

    if len(time_stamps) >= 2:
        print("\n" + "=" * 60)
        print(f"Total frames: {timed_frames}")
        print("=" * 60)

        time_deltas = [
            t2 - t1 for t2, t1 in zip(time_stamps[1:], time_stamps[:-1])
        ]
        avg_delta = sum(time_deltas) / len(time_deltas)
        avg_delta_ms = avg_delta * 1000
        throughput_fps = 1.0 / avg_delta if avg_delta > 0 else 0

        print(f"Frame-to-Frame Latency: {avg_delta_ms:0.2f} ms")
        print(f"Throughput FPS: {throughput_fps:0.2f}")
        print("=" * 60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FPS measurement tool (synthetic input)")
    parser.add_argument("-d", "--dfp", type=str, required=True, help="Path to DFP file")
    parser.add_argument(
        "-t", "--task", type=str, required=False,
        help="Task type (e.g. 'yolov8_det', 'yolov8_seg'). Required only when not using -p"
    )
    parser.add_argument(
        "-f", "--frames", type=int, default=1000, help="Max frames to process"
    )
    parser.add_argument(
        "-p", "--post_model", type=str, default=None,
        help="Path to the post-processing ONNX model (optional)"
    )
    args = parser.parse_args()

    if not args.post_model and not args.task:
        parser.error("Either --task or --post_model must be provided")

    run_async(
        dfp_path=args.dfp,
        task=args.task,
        post_model=args.post_model,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    main()
