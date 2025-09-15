import sys
import cv2
import argparse
import time
from collections import deque

from utils.vis_utils import draw_text, draw_bbox, draw_fps
from utils.logger_utils import setup_logger


WINDOW_NAME = "VisionStream"
RTSP_RETRY_ATTEMPTS = 3
RTSP_RETRY_DELAY = 2  # seconds
FPS_FRAME_COUNT = 60  # Number of frames for moving average
CONSEC_FAIL_MAX = 30  # for read failures

# Success and error codes
EXIT_OK = 0          
EXIT_INIT_FAIL = 1   
EXIT_RUNTIME_FAIL = 2  


def parse_args():
    parser = argparse.ArgumentParser(description="VisionStream - Real-time input handler")
    # Input controls (None = don't override YAML)
    parser.add_argument("--source", choices=["webcam", "file", "rtsp"], default="webcam", help="Input source")
    parser.add_argument("--path", type=str, help="File path (for file) or RTSP URL (for rtsp)")

    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO"], default="INFO",
                        help="Logging level (default: INFO)")

    # Overlay controls
    fps_group = parser.add_mutually_exclusive_group()
    fps_group.add_argument("--show-fps", dest="show_fps", action="store_true",
                           help="Force-show FPS overlay")
    fps_group.add_argument("--no-fps", dest="show_fps", action="store_false",
                           help="Hide FPS overlay")
    parser.set_defaults(show_fps=True)  # Default to True instead of None

    return parser.parse_args()


def open_capture(source, path, logger):
    if source == "webcam":
        cap = cv2.VideoCapture(0)

    elif source == "file":
        if not path:
            logger.error("Video file path is required. Use --path to specify the file location")
            return None
        cap = cv2.VideoCapture(path)

    elif source == "rtsp":
        if not path:
            logger.error("RTSP URL is required. Use --path to specify the stream URL")
            return None

        cap = None
        for attempt in range(1, RTSP_RETRY_ATTEMPTS + 1):
            logger.debug(f"Attempting to connect to RTSP stream (attempt {attempt}/{RTSP_RETRY_ATTEMPTS})...")
            cap = cv2.VideoCapture(path)

            if cap.isOpened():
                break
            else:
                cap.release()
                if attempt < RTSP_RETRY_ATTEMPTS:
                    logger.debug(f"Connection failed. Retrying in {RTSP_RETRY_DELAY} seconds...")
                    time.sleep(RTSP_RETRY_DELAY)
                else:
                    logger.error("All RTSP connection attempts failed")
                    return None

    else:
        logger.error("Invalid source type. Use 'webcam', 'file', or 'rtsp'")
        return None

    if cap is None or not cap.isOpened():
        logger.error(f"Failed to connect to {source}")
        if path:
            if source == "file":
                logger.error("Tips: verify file path, permissions, and that the file is not zero-length or locked.")
            elif source == "rtsp":
                logger.error("Tips: verify URL, credentials, network/firewall, and that the camera is online.")
        else:
            logger.error("Tips: ensure the webcam is connected, has drivers, and isn't used by another app.")
        return None


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Successfully connected to {source}" + (f": {path}" if path else " (default camera)"))
    logger.info(f"Stream resolution: {width}x{height}")
    return cap


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)


    # Resolve overlay settings
    show_fps = args.show_fps if args.show_fps is not None else True

    cap = open_capture(args.source, args.path, logger)
    if cap is None:
        logger.error("Unable to start VisionStream due to input error.")
        sys.exit(EXIT_INIT_FAIL)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    logger.info("VisionStream is running. Press 'q' to quit or close the window to exit.")

    frame_durations = deque(maxlen=FPS_FRAME_COUNT)
    prev_time = time.perf_counter()
    frame_count = 0
    start_time = prev_time

    consecutive_failures = 0  
    exit_code = EXIT_OK       

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1 
                # ------------------------------
                # Failure handling logic
                # ------------------------------
                if args.source == "file":
                    # For files, a single False often means EOF; exit calmly.
                    logger.info("Video file has reached the end")
                    exit_code = EXIT_OK
                    break

                if consecutive_failures >= CONSEC_FAIL_MAX:
                    logger.error(
                        f"Failed to read frames {CONSEC_FAIL_MAX} times in a row. "
                        "Likely causes: device unplugged, stream dropped, or permissions lost."
                    )
                    exit_code = EXIT_RUNTIME_FAIL
                    break

                # Small backoff helps avoid a hot loop if device is momentarily unavailable
                time.sleep(0.02)
                continue
            else:
                consecutive_failures = 0  # NEW (reset after a good frame)
            # ------------------------------
            # FPS calculation and overlay
            # ------------------------------
            curr_time = time.perf_counter()
            frame_duration = curr_time - prev_time
            frame_durations.append(frame_duration)
            prev_time = curr_time
            frame_count += 1

            if frame_durations:
                avg_duration = sum(frame_durations) / len(frame_durations)
                fps = 1.0 / avg_duration if avg_duration > 0 else 0.0
            else:
                fps = 0.0

            draw_text(frame, "VisionStream Demo", (10, 30))
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2
            bbox_size = 100
            draw_bbox(frame, (cx - bbox_size, cy - bbox_size, cx + bbox_size, cy + bbox_size), label="Dummy")
            if show_fps:
                draw_fps(frame, fps, pos=(10, 60))

            if args.log_level == "DEBUG" and frame_count % FPS_FRAME_COUNT == 0:
                logger.debug(f"Frame {frame_count}, size={w}x{h}, smoothed FPS={fps:.1f}")

            cv2.imshow(WINDOW_NAME, frame)
            # ------------------------------
            # Exit conditions
            # ------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User requested exit (q key)")
                exit_code = EXIT_OK
                break

            # If window is closed, stop loop
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed by user")
                exit_code = EXIT_OK
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received (Ctrl+C)")
        exit_code = EXIT_OK
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        exit_code = EXIT_RUNTIME_FAIL
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        total_time = time.perf_counter() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0.0
        logger.info(
            f"VisionStream closed. Frames={frame_count}, Duration={total_time:.2f}s, Avg FPS={avg_fps:.1f}"
        )
        sys.exit(exit_code)  # NEW


if __name__ == "__main__":
    main()
