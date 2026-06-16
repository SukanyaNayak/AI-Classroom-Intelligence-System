# config.py

CONFIDENCE_THRESHOLD = 0.4

# YOLO COCO class IDs
PERSON_CLASS = 0

# Teacher detection: define a region-of-interest for the front of class
# Format: (x_start_ratio, y_start_ratio, x_end_ratio, y_end_ratio)
# e.g. front 30% of frame width, full height
TEACHER_ZONE = (0.0, 0.0, 0.30, 1.0)

# Seat zone grid: divide classroom into rows x columns
# Each zone is (row, col) representing a seat position
# Format: zone_index -> (x_start_ratio, y_start_ratio, x_end_ratio, y_end_ratio)
SEAT_ZONES = {
    1: (0.30, 0.0, 0.50, 0.5),      # Row 1, Col 1
    2: (0.50, 0.0, 0.70, 0.5),      # Row 1, Col 2
    3: (0.70, 0.0, 0.90, 0.5),      # Row 1, Col 3
    4: (0.30, 0.5, 0.50, 1.0),      # Row 2, Col 1
    5: (0.50, 0.5, 0.70, 1.0),      # Row 2, Col 2
    6: (0.70, 0.5, 0.90, 1.0),      # Row 2, Col 3
}

# Instruments / accessories to detect (COCO class IDs)
INSTRUMENT_CLASSES = {
    24: "Bag",
    26: "Handbag",
    28: "Suitcase",
    63: "Laptop",
    64: "Mouse",
    66: "Keyboard",
    67: "Smartphone",
    73: "Book",
    39: "Bottle",
}

# Color per instrument for bounding box (BGR)
INSTRUMENT_COLORS = {
    "Bag":        (200, 120,  30),
    "Handbag":    (180, 100,  50),
    "Suitcase":   (160,  90,  20),
    "Laptop":     ( 30, 180, 255),
    "Mouse":      ( 80, 200, 180),
    "Keyboard":   ( 50, 160, 220),
    "Smartphone": (255,  60, 120),
    "Book":       ( 80, 200,  80),
    "Bottle":     (120, 220, 180),
}
