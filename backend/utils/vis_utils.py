import cv2

def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.7, thickness=2):
    """
    Draw simple text on a frame.
    Args:
        frame: image (OpenCV BGR)
        text: string
        pos: (x, y) position
    """
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )
    return frame

def draw_bbox(frame, bbox, label=None, color=(0, 255, 255), thickness=2):
    """
    Draw bounding box with optional label.
    Args:
        frame: image (OpenCV BGR)
        bbox: (x1, y1, x2, y2)
        label: optional string
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA
        )
    return frame

def draw_fps(frame, fps, pos=(10, 30), color=(0, 255, 0)):
    """
    Overlay FPS counter on frame.
    Args:
        frame: image (OpenCV BGR)
        fps: float
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        lineType=cv2.LINE_AA
    )
    return frame
