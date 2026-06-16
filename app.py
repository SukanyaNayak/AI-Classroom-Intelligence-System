import os
import asyncio
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from detector import detect_all
from analytics import update_heatmap, accessibility_score, detect_seat_occupancy, calculate_engagement_score
from logger import log_entry, log_seat_occupancy, log_engagement_summary, init_video_writer, write_frame_to_video, finalize_video_recording, get_user_recordings
from config import INSTRUMENT_CLASSES, SEAT_ZONES
from auth import login, register, change_password, get_all_users
from pose_detector import detect_raised_hands

# ── Must be FIRST streamlit call ──────────────────────────────
st.set_page_config(page_title="Classroom AI", layout="wide", page_icon="🎓")

# ── Windows asyncio fix ───────────────────────────────────────
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Session state ─────────────────────────────────────────────
if "user"              not in st.session_state: st.session_state.user              = None
if "running"           not in st.session_state: st.session_state.running           = False
if "frame_count"       not in st.session_state: st.session_state.frame_count       = 0
if "heatmap"           not in st.session_state: st.session_state.heatmap           = np.zeros((10, 10), dtype=float)
if "instrument_totals" not in st.session_state: st.session_state.instrument_totals = {n: 0 for n in INSTRUMENT_CLASSES.values()}
if "last_frame"        not in st.session_state: st.session_state.last_frame        = None
if "last_count"        not in st.session_state: st.session_state.last_count        = 0
if "last_teacher"      not in st.session_state: st.session_state.last_teacher      = False
if "last_instruments"  not in st.session_state: st.session_state.last_instruments  = {}
if "cap"               not in st.session_state: st.session_state.cap               = None
if "source_mode"       not in st.session_state: st.session_state.source_mode       = "Webcam"
if "raised_hands"      not in st.session_state: st.session_state.raised_hands      = []
if "hand_raise_log"    not in st.session_state: st.session_state.hand_raise_log    = []
if "pose_enabled"      not in st.session_state: st.session_state.pose_enabled      = False
if "last_notif_count"  not in st.session_state: st.session_state.last_notif_count  = 0
if "occupancy_history" not in st.session_state: st.session_state.occupancy_history = []
if "engagement_scores" not in st.session_state: st.session_state.engagement_scores = {}
if "video_writer"      not in st.session_state: st.session_state.video_writer      = None
if "video_path"        not in st.session_state: st.session_state.video_path        = None

# ─────────────────────────────────────────────────────────────
# AUTH GATE
# ─────────────────────────────────────────────────────────────
if st.session_state.user is None:
    st.title("🔐 Classroom AI — Login")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username  = st.text_input("Username")
            password  = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = login(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab_register:
        with st.form("register_form"):
            new_user  = st.text_input("Choose username")
            new_pass  = st.text_input("Choose password",  type="password")
            new_pass2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                if new_pass != new_pass2:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif register(new_user, new_pass):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already taken.")

    st.stop()

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
user = st.session_state.user

# ── Header ────────────────────────────────────────────────────
col_title, col_logout = st.columns([5, 1])
col_title.title("🎓 AI Classroom Intelligence System")
with col_logout:
    st.markdown(f"👤 **{user['username']}** `{user['role']}`")
    if st.button("Logout"):
        st.session_state.user    = None
        st.session_state.running = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    source = st.selectbox(
        "Input source",
        ["Webcam", "Upload video", "Upload image", "RTSP URL"]
    )
    st.session_state.source_mode = source

    conf        = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    st.session_state.pose_enabled = st.checkbox("✋ Detect raised hands", value=True)
    log_enabled = st.checkbox("Enable logging", True)

    rtsp_url = ""
    if source == "RTSP URL":
        rtsp_url = st.text_input("RTSP stream URL")

    # ── Image uploader ────────────────────────────────────────
    uploaded_img = None
    if source == "Upload image":
        st.divider()
        st.markdown("### 🖼️ Upload Image")
        uploaded_img = st.file_uploader(
            "Choose a classroom image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="img_uploader"
        )
        if uploaded_img is None:
            st.info("Upload a JPG / PNG image to run detection.")

    # ── Video uploader ────────────────────────────────────────
    uploaded_vid = None
    if source == "Upload video":
        st.divider()
        st.markdown("### 🎬 Upload Video")
        uploaded_vid = st.file_uploader(
            "Choose a classroom video",
            type=["mp4", "avi", "mov"],
            key="vid_uploader"
        )

    st.divider()
    st.markdown("**Teacher zone** = left 30% of frame")
    st.markdown("**Students** = rest of frame")

    # ── Admin user management ─────────────────────────────────
    if user["role"] == "admin":
        st.divider()
        st.subheader("👥 User Management")
        all_users = get_all_users()
        st.dataframe(pd.DataFrame(all_users).T)

        with st.expander("➕ Add new user"):
            with st.form("add_user"):
                nu  = st.text_input("Username")
                np_ = st.text_input("Password", type="password")
                nr  = st.selectbox("Role", ["teacher", "admin", "viewer"])
                if st.form_submit_button("Add"):
                    if register(nu, np_, nr):
                        st.success(f"User '{nu}' added.")
                    else:
                        st.error("Username already exists.")

    # ── Change password ───────────────────────────────────────
    st.divider()
    with st.expander("🔑 Change password"):
        with st.form("change_pw"):
            old_pw  = st.text_input("Current password", type="password")
            new_pw  = st.text_input("New password",     type="password")
            new_pw2 = st.text_input("Confirm new",      type="password")
            if st.form_submit_button("Update"):
                if new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif change_password(user["username"], old_pw, new_pw):
                    st.success("Password updated.")
                else:
                    st.error("Current password is wrong.")

# ── Controls ──────────────────────────────────────────────────
if source in ["Webcam", "Upload video", "RTSP URL"]:
    c1, c2, c3 = st.columns(3)

    if c1.button("▶ Start"):
        if source == "Upload video" and uploaded_vid:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_vid.read())
            if st.session_state.cap:
                st.session_state.cap.release()
            st.session_state.cap          = cv2.VideoCapture("temp_video.mp4")
            st.session_state.running      = True
            st.session_state.video_writer = None
            st.session_state.video_path   = None
        elif source == "Webcam":
            if st.session_state.cap:
                st.session_state.cap.release()
            st.session_state.cap          = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            st.session_state.running      = True
            st.session_state.video_writer = None
            st.session_state.video_path   = None
        elif source == "RTSP URL" and rtsp_url:
            if st.session_state.cap:
                st.session_state.cap.release()
            st.session_state.cap          = cv2.VideoCapture(rtsp_url)
            st.session_state.running      = True
            st.session_state.video_writer = None
            st.session_state.video_path   = None

    if c2.button("⏹ Stop"):
        st.session_state.running = False
        if st.session_state.video_writer:
            finalize_video_recording(st.session_state.video_writer)
            st.session_state.video_writer = None
            st.success(f"✅ Recording saved to: {st.session_state.video_path}")
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

    if c3.button("🔄 Reset logs"):
        for f in ["data/attendance_log.csv", "data/engagement_log.csv"]:
            if os.path.exists(f):
                os.remove(f)
        st.session_state.heatmap           = np.zeros((10, 10), dtype=float)
        st.session_state.instrument_totals = {n: 0 for n in INSTRUMENT_CLASSES.values()}
        st.session_state.occupancy_history = []
        st.session_state.engagement_scores = {}
        st.session_state.last_frame        = None
        st.session_state.frame_count       = 0
        st.success("Logs cleared.")

# ─────────────────────────────────────────────────────────────
# SHARED DETECTION FUNCTION — used by ALL 4 sources
# ─────────────────────────────────────────────────────────────
def run_detection(frame):
    from datetime import datetime

    # Step 1 — student / teacher / instrument
    annotated, student_count, teacher_present, \
        instrument_counts, detections, seat_occupancy = detect_all(frame, conf)

    # Step 2 — pose / raised hands
    if st.session_state.pose_enabled:
        annotated, raised_students = detect_raised_hands(annotated, conf=0.25)

        if len(raised_students) > st.session_state.last_notif_count:
            for s in raised_students:
                entry = {
                    "time":     datetime.now().strftime("%H:%M:%S"),
                    "student":  f"Student #{s['id']}",
                    "position": s["position"],
                    "row":      s["row"],
                    "reason":   s.get("reason", ""),
                }
                if entry not in st.session_state.hand_raise_log:
                    st.session_state.hand_raise_log.append(entry)

        st.session_state.raised_hands     = raised_students
        st.session_state.last_notif_count = len(raised_students)
    else:
        st.session_state.raised_hands     = []
        st.session_state.last_notif_count = 0

    # Step 3 — heatmap + instrument totals
    st.session_state.heatmap = update_heatmap(
        st.session_state.heatmap, detections, frame.shape)
    for k, v in instrument_counts.items():
        st.session_state.instrument_totals[k] += v

    # Step 4 — seat occupancy
    st.session_state.occupancy_history.append(seat_occupancy)
    log_seat_occupancy(seat_occupancy)

    # Step 5 — cache results
    st.session_state.last_frame       = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.session_state.last_count       = student_count
    st.session_state.last_teacher     = teacher_present
    st.session_state.last_instruments = dict(st.session_state.instrument_totals)
    st.session_state.frame_count     += 1

    # Step 6 — attendance log
    if log_enabled:
        log_entry(student_count, teacher_present, instrument_counts)

    return student_count, teacher_present, len(st.session_state.raised_hands)

# ─────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────
if source == "Upload image" and uploaded_img is not None:
    file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is not None:
        student_count, teacher_present, raised_count = run_detection(frame)
        st.session_state.running = False

        st.success(
            f"✅ Image analysed — **{student_count}** student(s), "
            f"**{raised_count}** hand(s) raised, "
            f"Teacher: **{'Present' if teacher_present else 'Absent'}**"
        )
        for s in st.session_state.raised_hands:
            st.warning(
                f"✋ **Student #{s['id']}** hand raised — "
                f"**{s['row']}**, **{s['position']}**"
                + (f" *(via: {s['reason']})*" if s.get("reason") else "")
            )
    else:
        st.error("Could not read the image. Please try a different file.")

# ─────────────────────────────────────────────────────────────
# RENDER RESULTS — shared by all sources
# ─────────────────────────────────────────────────────────────
def render_results():
    fc = st.session_state.frame_count
    col_feed, col_stats = st.columns([3, 2])

    with col_feed:
        st.subheader("📹 Detection Result"
                     if st.session_state.source_mode == "Upload image"
                     else "📹 Live Feed")

        if st.session_state.last_frame is not None:
            st.image(st.session_state.last_frame,
                     channels="RGB", width="stretch")
        else:
            if st.session_state.source_mode == "Upload image":
                st.info("Upload an image in the sidebar to run detection.")
            else:
                st.info("Press ▶ Start to begin detection.")

        # Hand raise alerts below feed
        raised = st.session_state.raised_hands
        if raised:
            st.markdown("---")
            st.markdown("### 🔔 Hand Raise Alerts")
            for student in raised:
                st.warning(
                    f"✋ **Student #{student['id']}** is raising their hand — "
                    f"**{student['row']}**, **{student['position']}**"
                )
        elif st.session_state.pose_enabled and \
             st.session_state.last_frame is not None:
            st.success("✅ No hands raised currently.")

    with col_stats:
        st.subheader("📊 Metrics")

        if st.session_state.last_frame is not None:
            if st.session_state.last_teacher:
                st.success("🟢 Teacher: **PRESENT**")
            else:
                st.error("🔴 Teacher: **ABSENT**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Students",     st.session_state.last_count)
            m2.metric("Hands raised", len(st.session_state.raised_hands))
            active_zones = sum(
                sum(row) for row in
                accessibility_score(st.session_state.heatmap)
            )
            m3.metric("Active zones", f"{active_zones}/100")

            if st.session_state.hand_raise_log:
                st.markdown("#### 📋 Hand raise history")
                log_df = pd.DataFrame(st.session_state.hand_raise_log)
                st.dataframe(log_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Download hand raise log",
                    log_df.to_csv(index=False),
                    "hand_raise_log.csv",
                    "text/csv",
                    key=f"dl_hand_{fc}"
                )

            active = {k: v for k, v in
                      st.session_state.last_instruments.items() if v > 0}
            if active:
                fig = px.bar(
                    x=list(active.keys()),
                    y=list(active.values()),
                    labels={"x": "Item", "y": "Detections"},
                    title="Instruments & accessories detected",
                    color=list(active.keys()),
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width="stretch", key=f"inst_{fc}")
            else:
                st.info("No instruments detected yet.")

# ─────────────────────────────────────────────────────────────
# LIVE FEED FRAGMENT — webcam / video / RTSP
# ─────────────────────────────────────────────────────────────
@st.fragment(run_every=0.05)
def live_feed():
    if st.session_state.source_mode == "Upload image":
        render_results()
        return

    cap = st.session_state.cap

    if st.session_state.running and cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            run_detection(frame)

            # Video writer
            fc = st.session_state.frame_count
            if st.session_state.video_writer is None and fc == 1:
                writer, video_path = init_video_writer(
                    st.session_state.user,
                    cv2.cvtColor(
                        st.session_state.last_frame,
                        cv2.COLOR_RGB2BGR
                    ).shape
                )
                st.session_state.video_writer = writer
                st.session_state.video_path   = video_path

            if st.session_state.video_writer:
                bgr = cv2.cvtColor(
                    st.session_state.last_frame, cv2.COLOR_RGB2BGR)
                write_frame_to_video(st.session_state.video_writer, bgr)

        else:
            st.session_state.running = False
            if st.session_state.video_writer:
                finalize_video_recording(st.session_state.video_writer)
                st.session_state.video_writer = None
            cap.release()
            st.session_state.cap = None

    render_results()

# ── Call fragment ─────────────────────────────────────────────
live_feed()

# ─────────────────────────────────────────────────────────────
# HEATMAP + ATTENDANCE HISTORY
# ─────────────────────────────────────────────────────────────
st.divider()
col_heat, col_hist = st.columns(2)

with col_heat:
    st.subheader("🗺️ Seat presence heatmap")
    fig_heat = px.imshow(
        st.session_state.heatmap,
        color_continuous_scale="YlOrRd",
        labels={"color": "Visits"}
    )
    fig_heat.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_heat, width="stretch", key="heatmap_static")

with col_hist:
    st.subheader("📈 Attendance history")
    try:
        df = pd.read_csv("data/attendance_log.csv")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        if df.empty:
            st.info("No data yet.")
        elif "student_count" not in df.columns:
            st.warning(f"Unexpected columns: {list(df.columns)}")
            st.dataframe(df)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            m1, m2, m3 = st.columns(3)
            m1.metric("Peak students",    int(df["student_count"].max()))
            m2.metric("Average students", round(float(df["student_count"].mean()), 1))
            m3.metric("Total entries",    df.index.nunique())
            st.line_chart(df["student_count"])
            if "teacher_present" in df.columns:
                absent_count = int((df["teacher_present"] == False).sum())
                if absent_count:
                    st.warning(f"⚠️ Teacher absent in {absent_count} recorded frames.")
            st.download_button("⬇ Download CSV",
                               df.to_csv(), "attendance.csv", "text/csv")
    except FileNotFoundError:
        st.info("No attendance data yet.")
    except pd.errors.EmptyDataError:
        st.info("Log file is empty.")

# ─────────────────────────────────────────────────────────────
# ENGAGEMENT SCORES
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("👥 Seat Zone Engagement Scores")

if st.session_state.frame_count > 0:
    engagement_scores = calculate_engagement_score(
        st.session_state.occupancy_history,
        st.session_state.frame_count
    )
    st.session_state.engagement_scores = engagement_scores

    engagement_cols = st.columns(3)
    for idx, (seat_id, score) in enumerate(sorted(engagement_scores.items())):
        with engagement_cols[idx % 3]:
            st.metric(f"Seat {seat_id}", f"{score:.1f}%")

    df_eng = pd.DataFrame({
        "Seat":         [f"Seat {s}" for s in sorted(engagement_scores.keys())],
        "Engagement %": [engagement_scores[s] for s in sorted(engagement_scores.keys())]
    })
    fig_eng = px.bar(df_eng, x="Seat", y="Engagement %",
                     title="Engagement Score by Seat Zone",
                     color="Engagement %",
                     color_continuous_scale="RdYlGn")
    st.plotly_chart(fig_eng, width="stretch", key="engagement_chart")

    if st.button("📊 Save Engagement Summary"):
        log_engagement_summary(engagement_scores)
        st.success("Engagement summary saved!")
else:
    st.info("No engagement data yet. Start detection to begin tracking.")

# ─────────────────────────────────────────────────────────────
# SESSION RECORDINGS
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🎥 Session Recordings")

if st.session_state.user:
    recordings = get_user_recordings(st.session_state.user)
    if recordings:
        st.write(f"**Found {len(recordings)} recording(s)**")
        rec_data = []
        for filename, filepath, size_mb in recordings:
            timestamp_str = filename.replace("session_", "").replace(".mp4", "")
            rec_data.append({
                "Session":   timestamp_str,
                "Size (MB)": size_mb,
                "File":      filename
            })
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True,
                     hide_index=True)

        selected_recording = st.selectbox(
            "Choose a recording to download:",
            [r[0] for r in recordings],
            label_visibility="collapsed"
        )
        if selected_recording:
            selected_path = next(
                (r[1] for r in recordings if r[0] == selected_recording), None)
            if selected_path and os.path.exists(selected_path):
                col1, col2 = st.columns(2)
                with col1:
                    with open(selected_path, "rb") as f:
                        st.download_button(
                            label=f"⬇ Download {selected_recording}",
                            data=f.read(),
                            file_name=selected_recording,
                            mime="video/mp4"
                        )
                with col2:
                    if st.button(f"🗑️ Delete {selected_recording}"):
                        os.remove(selected_path)
                        st.success(f"Deleted: {selected_recording}")
                        st.rerun()
    else:
        st.info("No recordings yet. Start a detection session to create a recording.")
else:
    st.info("Log in to view your session recordings.")