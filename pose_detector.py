# pose_detector.py
import cv2
import numpy as np
from ultralytics import YOLO

pose_model = YOLO("yolov8n-pose.pt")

# YOLOv8 pose keypoint indices
NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10


def _get_reference_y(kp, conf_thresh):
    """
    Returns the best available vertical reference point.
    Tries nose → ears → eyes → shoulders in order.
    More fallbacks = works in more camera angles.
    """
    candidates = [
        (NOSE,           kp[NOSE][2]),
        (LEFT_EAR,       kp[LEFT_EAR][2]),
        (RIGHT_EAR,      kp[RIGHT_EAR][2]),
        (LEFT_EYE,       kp[LEFT_EYE][2]),
        (RIGHT_EYE,      kp[RIGHT_EYE][2]),
    ]
    for idx, conf in candidates:
        if conf > conf_thresh:
            return kp[idx][1]

    # Fall back to average of shoulders
    ls_conf = kp[LEFT_SHOULDER][2]
    rs_conf = kp[RIGHT_SHOULDER][2]
    if ls_conf > conf_thresh and rs_conf > conf_thresh:
        return (kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2
    if ls_conf > conf_thresh:
        return kp[LEFT_SHOULDER][1]
    if rs_conf > conf_thresh:
        return kp[RIGHT_SHOULDER][1]

    return None  # cannot determine reference


def _get_shoulder_y(kp, conf_thresh):
    """Returns average shoulder Y for elbow comparison."""
    ls_conf = kp[LEFT_SHOULDER][2]
    rs_conf = kp[RIGHT_SHOULDER][2]
    if ls_conf > conf_thresh and rs_conf > conf_thresh:
        return (kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2
    if ls_conf > conf_thresh:
        return kp[LEFT_SHOULDER][1]
    if rs_conf > conf_thresh:
        return kp[RIGHT_SHOULDER][1]
    return None


def _is_hand_raised(kp, conf_thresh=0.25):
    """
    Multi-stage detection — passes if ANY condition is true:

    Stage 1 — Wrist above nose/ears/eyes (classic full raise)
    Stage 2 — Wrist above shoulders (partial / seated raise)
    Stage 3 — Elbow above shoulders (arm lifted even if wrist
               not visible — common when camera is far away)
    Stage 4 — Wrist above elbow significantly (elbow bent upward)
    """
    ref_y      = _get_reference_y(kp, conf_thresh)
    shoulder_y = _get_shoulder_y(kp, conf_thresh)

    lw_conf = kp[LEFT_WRIST][2]
    rw_conf = kp[RIGHT_WRIST][2]
    le_conf = kp[LEFT_ELBOW][2]
    re_conf = kp[RIGHT_ELBOW][2]

    lw_y = kp[LEFT_WRIST][1]
    rw_y = kp[RIGHT_WRIST][1]
    le_y = kp[LEFT_ELBOW][1]
    re_y = kp[RIGHT_ELBOW][1]

    # ── Stage 1: wrist clearly above head reference ───────────
    if ref_y is not None:
        if lw_conf > conf_thresh and lw_y < ref_y - 20:
            return True, "wrist above head"
        if rw_conf > conf_thresh and rw_y < ref_y - 20:
            return True, "wrist above head"

    # ── Stage 2: wrist above shoulders ───────────────────────
    if shoulder_y is not None:
        if lw_conf > conf_thresh and lw_y < shoulder_y - 15:
            return True, "wrist above shoulder"
        if rw_conf > conf_thresh and rw_y < shoulder_y - 15:
            return True, "wrist above shoulder"

    # ── Stage 3: elbow above shoulders (far camera) ──────────
    if shoulder_y is not None:
        if le_conf > conf_thresh and le_y < shoulder_y - 20:
            return True, "elbow above shoulder"
        if re_conf > conf_thresh and re_y < shoulder_y - 20:
            return True, "elbow above shoulder"

    # ── Stage 4: wrist significantly above its own elbow ─────
    if lw_conf > conf_thresh and le_conf > conf_thresh:
        if le_y - lw_y > 40:   # elbow is below wrist by 40px
            return True, "wrist above elbow"
    if rw_conf > conf_thresh and re_conf > conf_thresh:
        if re_y - rw_y > 40:
            return True, "wrist above elbow"

    return False, ""


def _draw_skeleton(frame, kp, conf_thresh, color=(0, 200, 100)):
    """Draws basic skeleton lines between keypoints."""
    connections = [
        (LEFT_SHOULDER,  LEFT_ELBOW),
        (LEFT_ELBOW,     LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW,    RIGHT_WRIST),
        (LEFT_SHOULDER,  RIGHT_SHOULDER),
    ]
    for a, b in connections:
        if kp[a][2] > conf_thresh and kp[b][2] > conf_thresh:
            pt1 = (int(kp[a][0]), int(kp[a][1]))
            pt2 = (int(kp[b][0]), int(kp[b][1]))
            cv2.line(frame, pt1, pt2, color, 2)

    # Draw joint circles
    for idx in [LEFT_SHOULDER, RIGHT_SHOULDER,
                LEFT_ELBOW,    RIGHT_ELBOW,
                LEFT_WRIST,    RIGHT_WRIST]:
        if kp[idx][2] > conf_thresh:
            pt = (int(kp[idx][0]), int(kp[idx][1]))
            cv2.circle(frame, pt, 5, color, -1)


def detect_raised_hands(frame, conf=0.25):
    """
    Runs pose estimation and detects raised hands.

    Lower default conf (0.25) works much better for:
    - Far away students
    - Partial occlusion
    - Side-angle cameras
    - Low lighting classrooms

    Returns:
        annotated_frame  - frame with skeleton + highlights
        raised_students  - list of dicts per raised hand
    """
    # Run at lower internal confidence for better keypoint coverage
    results   = pose_model(frame, conf=0.2, verbose=False)
    annotated = frame.copy()
    raised_students = []
    h, w = frame.shape[:2]

    for result in results:
        if result.keypoints is None:
            continue

        kps_data = result.keypoints.data
        boxes    = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for idx in range(len(kps_data)):
            kp  = kps_data[idx].cpu().numpy()
            box = boxes[idx].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Position labels
            position = ("Left side"  if cx < w // 3 else
                        "Center"     if cx < 2 * w // 3 else
                        "Right side")
            row      = ("Front row"  if cy < h // 3 else
                        "Middle row" if cy < 2 * h // 3 else
                        "Back row")

            raised, reason = _is_hand_raised(kp, conf)

            if raised:
                raised_students.append({
                    "id":       idx + 1,
                    "position": position,
                    "row":      row,
                    "reason":   reason,
                    "bbox":     (x1, y1, x2, y2),
                    "cx":       cx,
                    "cy":       cy,
                })

                # Yellow highlight box
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              (0, 255, 255), 3)

                # Label with reason
                label = f"Hand Raised! ({reason})"
                label_y = max(y1 - 12, 20)
                cv2.putText(annotated, label,
                            (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

                # Yellow skeleton
                _draw_skeleton(annotated, kp, conf,
                               color=(0, 255, 255))

                # Circle on visible wrists
                for wrist_idx in [LEFT_WRIST, RIGHT_WRIST]:
                    if kp[wrist_idx][2] > conf:
                        wx = int(kp[wrist_idx][0])
                        wy = int(kp[wrist_idx][1])
                        cv2.circle(annotated, (wx, wy),
                                   12, (0, 255, 255), -1)
                        cv2.circle(annotated, (wx, wy),
                                   12, (0, 200, 200), 2)
            else:
                # Normal green box + skeleton
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              (0, 200, 100), 1)
                _draw_skeleton(annotated, kp, conf,
                               color=(0, 200, 100))

    # Counter overlay
    count = len(raised_students)
    if count > 0:
        overlay_text = f"  Hands raised: {count}  "
        cv2.rectangle(annotated, (10, 55), (280, 90),
                      (0, 0, 0), -1)
        cv2.putText(annotated, overlay_text,
                    (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

    return annotated, raised_students