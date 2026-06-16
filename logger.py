# logger.py
import csv, os, cv2
from datetime import datetime

LOG_FILE = "data/attendance_log.csv"
ENGAGEMENT_LOG_FILE = "data/engagement_log.csv"
RECORDINGS_DIR = "data/recordings"

def init_video_writer(user_id, frame_shape, fps=30.0):
    """
    Initialize a video writer for a user session.
    
    Args:
        user_id: logged-in user identifier (str or dict with 'username' key)
        frame_shape: (height, width) of frames
        fps: frames per second
    
    Returns:
        (video_writer, video_path)
    """
    # Extract username if user_id is a dict
    if isinstance(user_id, dict):
        user_id = user_id.get("username", "unknown")
    
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    
    # Create user-specific subdirectory
    user_dir = os.path.join(RECORDINGS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(user_dir, f"session_{timestamp}.mp4")
    
    # Initialize VideoWriter (MP4 codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    return writer, video_path


def write_frame_to_video(writer, frame):
    """Write an annotated frame to the video file."""
    if writer and writer.isOpened():
        writer.write(frame)


def finalize_video_recording(writer):
    """Close and finalize the video recording."""
    if writer:
        writer.release()


def get_user_recordings(user_id):
    """
    Get list of all recordings for a user.
    
    Args:
        user_id: user identifier (str or dict with 'username' key)
    
    Returns:
        list of (filename, filepath, file_size_mb)
    """
    # Extract username if user_id is a dict
    if isinstance(user_id, dict):
        user_id = user_id.get("username", "unknown")
    
    user_dir = os.path.join(RECORDINGS_DIR, user_id)
    if not os.path.exists(user_dir):
        return []
    
    recordings = []
    for filename in os.listdir(user_dir):
        if filename.endswith(".mp4"):
            filepath = os.path.join(user_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            recordings.append((filename, filepath, round(size_mb, 2)))
    
    return sorted(recordings, reverse=True)  # Most recent first

def log_entry(student_count, teacher_present, instrument_counts):
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["timestamp", "student_count", "teacher_present"] + \
                     list(instrument_counts.keys())
            writer.writerow(header)
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            student_count,
            teacher_present
        ] + list(instrument_counts.values())
        writer.writerow(row)


def log_seat_occupancy(seat_occupancy):
    """Log seat zone occupancy for each frame."""
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(ENGAGEMENT_LOG_FILE) and os.path.getsize(ENGAGEMENT_LOG_FILE) > 0

    with open(ENGAGEMENT_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            seat_ids = sorted(seat_occupancy.keys())
            header = ["timestamp"] + [f"seat_{sid}" for sid in seat_ids]
            writer.writerow(header)
        
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        for seat_id in sorted(seat_occupancy.keys()):
            row.append(seat_occupancy[seat_id])
        writer.writerow(row)


def log_engagement_summary(engagement_scores):
    """Log final engagement scores for the session."""
    os.makedirs("data", exist_ok=True)
    summary_file = "data/engagement_summary.csv"
    file_exists = os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0

    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            seat_ids = sorted(engagement_scores.keys())
            header = ["session_timestamp"] + [f"seat_{sid}_engagement" for sid in seat_ids]
            writer.writerow(header)
        
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        for seat_id in sorted(engagement_scores.keys()):
            row.append(engagement_scores[seat_id])
        writer.writerow(row)