import numpy as np
from config import SEAT_ZONES

def update_heatmap(heatmap, detections, frame_shape):
    """Accumulates a presence heatmap from detected bounding box centers."""
    h, w = frame_shape[:2]
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # normalize to heatmap grid (e.g. 10x10)
        gx = min(int(cx / w * 10), 9)
        gy = min(int(cy / h * 10), 9)
        heatmap[gy][gx] += 1
    return heatmap

def accessibility_score(heatmap):
    """Returns which grid zones are accessible (used frequently)."""
    threshold = heatmap.max() * 0.3
    return (heatmap > threshold).tolist()


def detect_seat_occupancy(detections, frame_shape):
    """
    Detects which seat zones have students present.
    
    Returns:
        dict: {zone_id: occupancy (0 or 1)}
    """
    h, w = frame_shape[:2]
    occupancy = {zone_id: 0 for zone_id in SEAT_ZONES}
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        for zone_id, zone_bounds in SEAT_ZONES.items():
            x0 = zone_bounds[0] * w
            y0 = zone_bounds[1] * h
            x1_zone = zone_bounds[2] * w
            y1_zone = zone_bounds[3] * h
            
            if x0 <= cx <= x1_zone and y0 <= cy <= y1_zone:
                occupancy[zone_id] = 1
    
    return occupancy


def calculate_engagement_score(occupancy_history, total_duration_frames):
    """
    Calculate engagement score per seat zone.
    Score = (frames_occupied / total_frames) * 100
    
    Args:
        occupancy_history: list of occupancy dicts from each frame
        total_duration_frames: total number of frames in session
    
    Returns:
        dict: {zone_id: engagement_score (0-100)}
    """
    if not occupancy_history or total_duration_frames == 0:
        return {zone_id: 0 for zone_id in SEAT_ZONES}
    
    engagement = {}
    for zone_id in SEAT_ZONES:
        occupied_frames = sum(1 for occ in occupancy_history if occ.get(zone_id, 0) == 1)
        score = (occupied_frames / total_duration_frames) * 100
        engagement[zone_id] = round(score, 2)
    
    return engagement