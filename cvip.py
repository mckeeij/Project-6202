import argparse
import sys
from typing import Any, Dict, List, Set, Tuple

try:
    import cv2
except ImportError:
    print(
        "OpenCV is not installed. Run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    raise SystemExit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "Ultralytics is not installed. Run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-resolution webcam object detection demo."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model to use. yolov8n.pt is fastest, yolov8s.pt more accurate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=28,
        help="Camera capture width (lower = faster).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=28,
        help="Camera capture height (lower = faster).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Internal detector size. Smaller = faster (320-416 recommended).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (higher = fewer detections = faster).",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.60,
        help="Higher threshold required before a detection is called a person.",
    )
    parser.add_argument(
        "--display-scale",
        type=int,
        default=5,
        help="How much to enlarge the tiny camera image for viewing.",
    )
    parser.add_argument(
        "--exit-frames",
        type=int,
        default=10,
        help="How many missed frames before an object is considered gone.",
    )
    return parser.parse_args()


def collect_detections(result, person_conf: float) -> List[Dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    xyxy_boxes = boxes.xyxy.tolist()

    for class_id, confidence, xyxy in zip(class_ids, confidences, xyxy_boxes):
        label = result.names[int(class_id)]
        if label == "person" and confidence < person_conf:
            continue
        detections.append(
            {
                "class_id": int(class_id),
                "label": label,
                "confidence": confidence,
                "xyxy": [int(value) for value in xyxy],
            }
        )

    # Sort by confidence and limit to top 10 detections for performance
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections[:10]


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    return (
        80 + (class_id * 70) % 176,
        80 + (class_id * 130) % 176,
        80 + (class_id * 200) % 176,
    )


def draw_detections(frame, detections: List[Dict[str, Any]]):
    annotated = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["xyxy"]
        label = f'{format_label(item["label"])} {item["confidence"]:.2f}'
        color = color_for_class(item["class_id"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_origin = (x1, max(20, y1 - 8))
        cv2.putText(
            annotated,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    args = parse_args()
    visible_labels: Set[str] = set()
    missed_frames: Dict[str, int] = {}

    try:
        model = YOLO(args.model)
        # Simple warmup with a small dummy image
        import numpy as np
        dummy_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        model.predict(dummy_frame, imgsz=args.imgsz, conf=args.conf, verbose=False, device="cpu")
        print("Model loaded and warmed up", flush=True)
    except Exception as exc:
        print(
            "Could not load the YOLO model. On first run, Ultralytics usually "
            "downloads the model from the internet.",
            file=sys.stderr,
        )
        print(f"Model error: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(
            f"Could not open camera index {args.camera}. "
            "Try --camera 1 if you have multiple cameras.",
            file=sys.stderr,
        )
        return 1

    print("Console event logging started.", flush=True)
    print(f"Using model: {args.model}", flush=True)
    print("Press Q to quit.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the camera.", file=sys.stderr)
            break

        # Force a known low-resolution frame even if the camera ignores the set() calls.
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        try:
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
                device="cpu",
            )
        except Exception as exc:
            print(f"Detection failed: {exc}", file=sys.stderr)
            break

        detections = collect_detections(results[0], args.person_conf)
        current_labels = {item["label"] for item in detections}

        for label in sorted(current_labels):
            missed_frames[label] = 0
            if label not in visible_labels:
                print(f"Entered Frame: {format_label(label)}", flush=True)
                visible_labels.add(label)

        for label in list(visible_labels):
            if label in current_labels:
                continue
            missed_frames[label] = missed_frames.get(label, 0) + 1
            if missed_frames[label] >= args.exit_frames:
                print(f"Exited Frame: {format_label(label)}", flush=True)
                visible_labels.remove(label)
                missed_frames.pop(label, None)

        annotated = draw_detections(frame, detections)
        # Smaller display for better performance
        display = cv2.resize(
            annotated,
            (400, 300),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imshow("Object Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

