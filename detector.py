# detector.py
import os
import asyncio

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import cv2
from ultralytics import YOLO
from config import (CONFIDENCE_THRESHOLD, PERSON_CLASS,
                    TEACHER_ZONE, INSTRUMENT_CLASSES, INSTRUMENT_COLORS, SEAT_ZONES)
from analytics import detect_seat_occupancy

model = YOLO("yolov8n.pt")  # single shared model for all detections


def _is_in_zone(cx, cy, frame_w, frame_h, zone):
    """Check if a center point falls inside a ratio-defined zone."""
    x0 = zone[0] * frame_w
    y0 = zone[1] * frame_h
    x1 = zone[2] * frame_w
    y1 = zone[3] * frame_h
    return x0 <= cx <= x1 and y0 <= cy <= y1


def detect_all(frame, conf=CONFIDENCE_THRESHOLD):
    """
    Single YOLO pass — returns:
      annotated_frame, student_count, teacher_present,
      instrument_counts (dict), detections (raw boxes), seat_occupancy (dict)
    """
    h, w = frame.shape[:2]
    results = model(frame, conf=conf, verbose=False)
    boxes = results[0].boxes

    annotated = frame.copy()
    student_count = 0
    teacher_present = False
    instrument_counts = {name: 0 for name in INSTRUMENT_CLASSES.values()}
    student_detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ── Person detection ──────────────────────────────────
        if cls_id == PERSON_CLASS:
            in_teacher_zone = _is_in_zone(cx, cy, w, h, TEACHER_ZONE)

            if in_teacher_zone:
                # Draw teacher box in blue
                teacher_present = True
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
                cv2.putText(annotated, f"Teacher {conf_val:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 140, 0), 2)
            else:
                # Draw student box in green
                student_count += 1
                student_detections.append(box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 2)
                cv2.putText(annotated, f"Student {conf_val:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 200, 100), 1)

        # ── Instrument detection ──────────────────────────────
        elif cls_id in INSTRUMENT_CLASSES:
            label = INSTRUMENT_CLASSES[cls_id]
            color = INSTRUMENT_COLORS.get(label, (200, 200, 200))
            instrument_counts[label] += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {conf_val:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

    # ── Overlays ──────────────────────────────────────────────
    # Teacher zone boundary line
    zone_x = int(TEACHER_ZONE[2] * w)
    cv2.line(annotated, (zone_x, 0), (zone_x, h), (255, 140, 0), 1)
    cv2.putText(annotated, "Teacher zone",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 140, 0), 1)

    # Seat zone boundaries
    for zone_id, bounds in SEAT_ZONES.items():
        x0 = int(bounds[0] * w)
        y0 = int(bounds[1] * h)
        x1_zone = int(bounds[2] * w)
        y1_zone = int(bounds[3] * h)
        cv2.rectangle(annotated, (x0, y0), (x1_zone, y1_zone), (100, 100, 255), 1)
        cv2.putText(annotated, f"Seat {zone_id}",
                    (x0 + 5, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 100, 255), 1)

    # Student count top-right
    cv2.putText(annotated, f"Students: {student_count}",
                (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 220, 100), 2)

    # Teacher status banner
    status_text = "Teacher: PRESENT" if teacher_present else "Teacher: ABSENT"
    status_color = (255, 140, 0) if teacher_present else (0, 60, 255)
    cv2.putText(annotated, status_text,
                (w - 260, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, status_color, 2)

    # Detect seat occupancy
    seat_occupancy = detect_seat_occupancy(student_detections, frame.shape)

    return annotated, student_count, teacher_present, instrument_counts, student_detections, seat_occupancy