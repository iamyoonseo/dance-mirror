"""
Dance Mirror — real-time dance coaching with pose comparison, beat scoring,
and AI feedback.

Usage:
    python app.py <video.mp4>
    python app.py <YouTube URL>

Controls:
    q  quit
    r  restart video
    p  pause / resume
    c  request AI coach feedback
"""

import sys, os, time, threading, subprocess
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[WARN] librosa not found — rhythm scoring disabled")

try:
    import ollama as _ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False
    print("[WARN] ollama not found — AI coach disabled")

try:
    import yt_dlp
    YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

# ── MediaPipe Tasks setup ──────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
OLLAMA_MODEL = "llama3.2"

BaseOptions      = mp.tasks.BaseOptions
PoseLandmarker   = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
RunningMode      = mp.tasks.vision.RunningMode

def make_detector():
    opts = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(opts)

_detect_ts = 0   # shared monotonic timestamp counter (ms)
_detect_lock = threading.Lock()

def detect(detector, rgb_frame):
    """Run pose detection using VIDEO mode with monotonic timestamps."""
    global _detect_ts
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    with _detect_lock:
        _detect_ts += 33   # ~30fps in ms
        ts = _detect_ts
    result = detector.detect_for_video(mp_img, ts)
    if result.pose_landmarks:
        return result.pose_landmarks[0]
    return None

# ── Landmark indices (MediaPipe 33-point body model) ──────────────────────────
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28

KEY_LM = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
          L_WRIST, R_WRIST, L_HIP, R_HIP,
          L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

LM_NAMES = {
    L_SHOULDER: "Left shoulder",  R_SHOULDER: "Right shoulder",
    L_ELBOW:    "Left elbow",     R_ELBOW:    "Right elbow",
    L_WRIST:    "Left wrist",     R_WRIST:    "Right wrist",
    L_HIP:      "Left hip",       R_HIP:      "Right hip",
    L_KNEE:     "Left knee",      R_KNEE:     "Right knee",
    L_ANKLE:    "Left ankle",     R_ANKLE:    "Right ankle",
}

BODY_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),   (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

# ── Constants ──────────────────────────────────────────────────────────────────
DISPLAY_H        = 480
SCORE_SMOOTH     = 20
AUDIO_COOLDOWN   = 3.5
POSE_THRESHOLD   = 0.30
VELOCITY_THRESH  = 0.06
BEAT_WINDOW      = 0.18
COACH_DISPLAY_SEC = 12

# ── Audio cue (macOS say) ──────────────────────────────────────────────────────
_last_cue: dict = {}
_speak_lock = threading.Lock()

def speak(message: str, key: str) -> None:
    now = time.time()
    if now - _last_cue.get(key, 0) < AUDIO_COOLDOWN:
        return
    _last_cue[key] = now
    def _run():
        with _speak_lock:
            subprocess.run(["say", "-r", "210", message], capture_output=True)
    threading.Thread(target=_run, daemon=True).start()

# ── Pose math ──────────────────────────────────────────────────────────────────

def normalise(landmarks):
    """
    landmarks: list of NormalizedLandmark (from new MediaPipe Tasks API)
    Returns (flat_vec, coord_dict) normalised to hip-centre + torso scale.
    y is image-space (positive = downward), so raised body parts → negative y.
    """
    if landmarks is None:
        return None, None

    def xy(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y], dtype=np.float32)

    hip   = (xy(L_HIP) + xy(R_HIP)) / 2.0
    scale = np.linalg.norm((xy(L_SHOULDER) + xy(R_SHOULDER)) / 2.0 - hip) + 1e-6

    coords, flat = {}, []
    for idx in KEY_LM:
        n = (xy(idx) - hip) / scale
        coords[idx] = n
        flat.extend(n)

    return np.array(flat, dtype=np.float32), coords


def mirror_vec(v: np.ndarray) -> np.ndarray:
    """Swap left/right pairs and negate x in the flat vector."""
    m = v.copy()
    for i in range(0, len(KEY_LM) - 1, 2):
        lx, ly = i*2, i*2+1
        rx, ry = (i+1)*2, (i+1)*2+1
        m[lx], m[rx] = -v[rx], -v[lx]
        m[ly], m[ry] =  v[ry],  v[ly]
    return m


def cosine_pct(a, b) -> float:
    if a is None or b is None:
        return 0.0
    d = np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
    return float(np.clip((np.dot(a, b) / d + 1) / 2 * 100, 0, 100))


def best_pose_score(ref, you) -> float:
    if ref is None or you is None:
        return 0.0
    return max(cosine_pct(ref, you), cosine_pct(mirror_vec(ref), you))

# ── Beat detection ─────────────────────────────────────────────────────────────

def load_beats(video_path: str):
    if not LIBROSA_OK:
        return [], 0
    try:
        print("Analysing audio for beat detection ...", end=" ", flush=True)
        y, sr = librosa.load(video_path, mono=True, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        print(f"done  ({bpm:.1f} BPM, {len(beat_times)} beats)")
        return beat_times, bpm
    except Exception as e:
        print(f"failed ({e})")
        return [], 0

# ── Rhythm tracker ─────────────────────────────────────────────────────────────

class RhythmTracker:
    def __init__(self, beat_times):
        self.beat_times   = np.array(beat_times)
        self.next_beat_i  = 0
        self.hits         = []
        self.prev_coords  = None
        self.last_velocity = 0.0
        self.beat_flash   = 0.0

    def update(self, current_sec: float, coords_you) -> float:
        velocity = 0.0
        if self.prev_coords is not None and coords_you is not None:
            for idx in KEY_LM:
                if idx in coords_you and idx in self.prev_coords:
                    velocity += float(np.linalg.norm(
                        coords_you[idx] - self.prev_coords[idx]))
        self.prev_coords   = coords_you
        self.last_velocity = velocity

        if len(self.beat_times) and self.next_beat_i < len(self.beat_times):
            bt = self.beat_times[self.next_beat_i]
            if current_sec > bt + BEAT_WINDOW:
                self.hits.append(velocity > VELOCITY_THRESH)
                self.next_beat_i += 1
                self.beat_flash = 0.25

        if self.beat_flash > 0:
            self.beat_flash -= 1/30

        return velocity

    def score(self) -> float:
        if not self.hits:
            return 0.0
        return sum(self.hits) / len(self.hits) * 100

    def reset(self, beat_times):
        self.__init__(beat_times)

# ── Session tracker ────────────────────────────────────────────────────────────

class SessionTracker:
    def __init__(self):
        self.pose_scores  = []
        self.joint_errors = defaultdict(list)
        self.start        = time.time()
        self.coach_lines  = []
        self.coach_until  = 0.0
        self.coach_pending = False

    def update(self, instant_score, coords_ref, coords_you):
        self.pose_scores.append(instant_score)
        if coords_ref and coords_you:
            for idx in KEY_LM:
                if idx in coords_ref and idx in coords_you:
                    d = float(np.linalg.norm(coords_you[idx] - coords_ref[idx]))
                    self.joint_errors[idx].append(d)

    def avg_score(self) -> float:
        return float(np.mean(self.pose_scores)) if self.pose_scores else 0.0

    def worst_joints(self, n=3):
        avgs = {lm: float(np.mean(v)) for lm, v in self.joint_errors.items() if v}
        return sorted(avgs.items(), key=lambda x: x[1], reverse=True)[:n]

    def duration(self) -> float:
        return time.time() - self.start

# ── AI coach (Ollama) ──────────────────────────────────────────────────────────

def request_coaching(session: SessionTracker, rhythm: RhythmTracker, on_done):
    if not OLLAMA_OK:
        on_done(["AI coach unavailable — run: pip install ollama"])
        return
    if session.coach_pending:
        return
    session.coach_pending = True

    def _call():
        try:
            worst = session.worst_joints(3)
            worst_str = ", ".join(
                f"{LM_NAMES.get(lm, lm)} ({err:.2f})" for lm, err in worst
            ) or "none identified"

            prompt = f"""You are an enthusiastic dance coach giving feedback after a practice run.

Student stats:
- Pose accuracy: {session.avg_score():.1f}%
- Rhythm accuracy: {rhythm.score():.1f}%
- Body parts needing most work: {worst_str}
- Duration: {session.duration():.0f}s

Give exactly 3 short coaching tips (1 sentence each). Be specific, positive, and actionable. No bullet symbols, no numbering — just 3 plain sentences separated by newlines."""

            resp = _ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp["message"]["content"].strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()][:3]
        except Exception as e:
            lines = [f"Coach error: {e}",
                     "Make sure Ollama is running: ollama serve",
                     f"And model is pulled: ollama pull {OLLAMA_MODEL}"]

        session.coach_pending = False
        on_done(lines)

    threading.Thread(target=_call, daemon=True).start()

# ── Drawing ────────────────────────────────────────────────────────────────────

def _joint_color(dist: float):
    t = min(dist / (POSE_THRESHOLD * 2), 1.0)
    return (0, int(200*(1-t)), int(220*t))   # BGR green→red


def draw_skeleton(frame, landmarks, coords_you=None, coords_ref=None):
    """
    Draw skeleton on frame.
    If coords_you and coords_ref provided: colour joints by per-joint deviation.
    Otherwise: flat cyan (reference style).
    """
    if landmarks is None:
        return

    h, w = frame.shape[:2]

    def pt(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    # Connections
    for a, b in BODY_CONNECTIONS:
        if landmarks[a].visibility > 0.3 and landmarks[b].visibility > 0.3:
            if coords_you and coords_ref:
                da = np.linalg.norm(coords_you.get(a, 0) - coords_ref.get(a, 0)) if a in coords_you and a in coords_ref else 0
                db = np.linalg.norm(coords_you.get(b, 0) - coords_ref.get(b, 0)) if b in coords_you and b in coords_ref else 0
                color = tuple(int(x) for x in (np.array(_joint_color(da)) + np.array(_joint_color(db))) // 2)
                thickness = 6
            else:
                color  = (200, 200, 0)   # cyan-ish for reference
                thickness = 4
            cv2.line(frame, pt(a), pt(b), color, thickness)

    # Joints
    for idx in KEY_LM:
        if landmarks[idx].visibility < 0.3:
            continue
        if coords_you and coords_ref and idx in coords_you and idx in coords_ref:
            dist  = float(np.linalg.norm(coords_you[idx] - coords_ref[idx]))
            color = _joint_color(dist)
            radius = 14
        else:
            color  = (0, 220, 220)
            radius = 10
        cv2.circle(frame, pt(idx), radius, color, -1)
        cv2.circle(frame, pt(idx), radius, (255, 255, 255), 2)


def _score_color(pct: float):
    t = pct / 100
    return (0, int(200*t), int(220*(1-t)))


def draw_hud(canvas, pose_smooth, pose_instant, rhythm_pct,
             beat_flash, tempo_bpm, coach_lines, coach_until):
    h, w = canvas.shape[:2]

    def score_bar(label, pct, by):
        bw = int(w * 0.68)
        bx = (w - bw) // 2
        bh = 18
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), (30,30,30), -1)
        fw = max(0, int(bw * pct / 100))
        if fw:
            cv2.rectangle(canvas, (bx, by), (bx+fw, by+bh), _score_color(pct), -1)
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), (160,160,160), 2)
        cv2.putText(canvas, label, (bx, by-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230,230,230), 2)
        return bx, bw

    bx, bw = score_bar(
        f"Pose  {pose_smooth:5.1f}%   (instant {pose_instant:5.1f}%)",
        pose_smooth, h-88)

    bpm_str = f"  {tempo_bpm:.0f} BPM" if tempo_bpm else ""
    score_bar(f"Rhythm {rhythm_pct:5.1f}%{bpm_str}", rhythm_pct, h-48)

    # Beat flash dot
    if beat_flash > 0:
        alpha = min(beat_flash / 0.25, 1.0)
        cx = bx + bw + 18
        cv2.circle(canvas, (cx, h-48+9), 10,
                   (int(50*alpha), int(220*alpha), int(220*alpha)), -1)

    # Coach overlay
    if coach_lines and time.time() < coach_until:
        lh2, pad = 26, 12
        box_h = pad*2 + lh2*(len(coach_lines)+1)
        box_x, box_y = 30, (h - box_h)//2 - 20
        box_w = w - 60
        overlay = canvas.copy()
        cv2.rectangle(overlay, (box_x, box_y),
                      (box_x+box_w, box_y+box_h), (10,10,10), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.putText(canvas, "AI Coach", (box_x+pad, box_y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)
        for i, line in enumerate(coach_lines):
            cv2.putText(canvas, line, (box_x+pad, box_y+pad+lh2*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240,240,240), 1)


def add_label(frame, text, color):
    cv2.putText(frame, text, (12,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
    cv2.putText(frame, text, (12,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,  2)

# ── Audio pose cues ────────────────────────────────────────────────────────────

def check_pose_cues(coords_ref, coords_you):
    if not coords_ref or not coords_you:
        return

    def dy(idx):
        return float(coords_ref[idx][1] - coords_you[idx][1])
    def dx(idx):
        return float(coords_ref[idx][0] - coords_you[idx][0])

    T = POSE_THRESHOLD
    checks = [
        (dy(L_WRIST) >  T, "Left arm higher!",     "la_up"),
        (dy(L_WRIST) < -T, "Left arm lower!",      "la_dn"),
        (dy(R_WRIST) >  T, "Right arm higher!",    "ra_up"),
        (dy(R_WRIST) < -T, "Right arm lower!",     "ra_dn"),
        (dx(L_WRIST) >  T, "Extend your left arm!", "la_out"),
        (dx(L_WRIST) < -T, "Bring left arm in!",   "la_in"),
        (dx(R_WRIST) >  T, "Bring right arm in!",  "ra_in"),
        (dx(R_WRIST) < -T, "Extend right arm!",    "ra_out"),
        (dy(L_KNEE)  >  T, "Bend your left knee!", "lk"),
        (dy(R_KNEE)  >  T, "Bend your right knee!","rk"),
    ]
    for condition, message, key in checks:
        if condition:
            speak(message, key)
            break

# ── Video download ─────────────────────────────────────────────────────────────

def resolve_video(source: str) -> str:
    if not source.startswith("http"):
        return source
    if not YTDLP_OK:
        sys.exit("[ERROR] yt-dlp not installed. Run: pip install yt-dlp")

    # Cache downloads in the project folder so we don't re-download each run
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    import hashlib
    url_hash  = hashlib.md5(source.encode()).hexdigest()[:10]
    cached    = os.path.join(cache_dir, f"{url_hash}.mp4")
    if os.path.exists(cached):
        print(f"Using cached video: {cached}")
        return cached

    out_dir  = cache_dir
    out_tmpl = os.path.join(out_dir, f"{url_hash}.%(ext)s")
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "merge_output_format": "mp4",
    }
    print(f"Downloading: {source}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
        path = ydl.prepare_filename(info)
        for ext in (".webm", ".mkv"):
            path = path.replace(ext, ".mp4")
    if not os.path.exists(path):
        for f in os.listdir(out_dir):
            candidate = os.path.join(out_dir, f)
            if url_hash in f:
                path = candidate; break
    print(f"Saved to: {path}")
    return path

# ── Main ───────────────────────────────────────────────────────────────────────

def main(source: str):
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[ERROR] Pose model not found at {MODEL_PATH}\n"
                 "Run: python -c \"import urllib.request; "
                 "urllib.request.urlretrieve('https://storage.googleapis.com/"
                 "mediapipe-models/pose_landmarker/pose_landmarker_lite/"
                 "float16/1/pose_landmarker_lite.task', 'pose_landmarker_lite.task')\"")

    video_path = resolve_video(source)

    cap_vid = cv2.VideoCapture(video_path)
    cap_cam = cv2.VideoCapture(0)
    if not cap_vid.isOpened(): sys.exit(f"[ERROR] Cannot open: {video_path}")
    if not cap_cam.isOpened(): sys.exit("[ERROR] Cannot open webcam")

    video_fps = cap_vid.get(cv2.CAP_PROP_FPS) or 30.0
    video_w   = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h   = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w   = int(video_w * DISPLAY_H / video_h)
    cam_panel_w = int(DISPLAY_H * 9 / 16)   # webcam shown at 9:16

    beat_times, tempo_bpm = load_beats(video_path)
    rhythm  = RhythmTracker(beat_times)
    session = SessionTracker()

    score_hist  = deque(maxlen=SCORE_SMOOTH)
    paused      = False
    frame_vid   = None
    video_start = None   # wall-clock time when video started (for frame sync)
    TARGET_FPS  = 30
    frame_dur   = 1.0 / TARGET_FPS
    next_frame  = time.time()

    print("Creating pose detectors ...")
    det_vid = make_detector()
    det_cam = make_detector()
    print("Ready.\n")
    print("Dance Mirror —  q: quit   r: restart   p: pause   c: AI coach\n")

    def on_coach_done(lines):
        session.coach_lines = lines
        session.coach_until = time.time() + COACH_DISPLAY_SEC
        speak(". ".join(lines), "coach")

    while True:
        # ── Webcam ─────────────────────────────────────────────────────────────
        ret_w, raw = cap_cam.read()
        if not ret_w: break
        frame_cam = cv2.flip(raw, 1)

        # ── Dance video (wall-clock sync to avoid slow-motion) ─────────────────
        if not paused:
            if video_start is None:
                video_start = time.time()
            # Seek to the frame that matches real elapsed time
            elapsed = time.time() - video_start
            target_frame = int(elapsed * video_fps)
            cap_vid.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret_v, frame_vid = cap_vid.read()
            if not ret_v:
                cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                score_hist.clear()
                rhythm.reset(beat_times)
                video_start = None
                continue

        current_sec = cap_vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ── Pose detection (parallel + downscaled for speed) ────────────────────
        DETECT_W = 640
        def prep(bgr):
            h, w = bgr.shape[:2]
            small = cv2.resize(bgr, (DETECT_W, int(h * DETECT_W / w)))
            return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_vid = ex.submit(detect, det_vid, prep(frame_vid))
            fut_cam = ex.submit(detect, det_cam, prep(frame_cam))
            lm_vid  = fut_vid.result()
            lm_cam  = fut_cam.result()

        vec_ref, coords_ref = normalise(lm_vid)
        vec_you, coords_you = normalise(lm_cam)

        # ── Scores ──────────────────────────────────────────────────────────────
        instant = best_pose_score(vec_ref, vec_you)
        score_hist.append(instant)
        smoothed = float(np.mean(score_hist)) if score_hist else 0.0

        if not paused:
            rhythm.update(current_sec, coords_you)

        session.update(instant, coords_ref, coords_you)
        check_pose_cues(coords_ref, coords_you)

        # ── Draw ────────────────────────────────────────────────────────────────
        draw_skeleton(frame_vid, lm_vid)                                   # reference: cyan
        draw_skeleton(frame_cam, lm_cam, coords_you, coords_ref)           # you: colour-coded

        ref_panel = cv2.resize(frame_vid, (panel_w, DISPLAY_H))
        you_panel = cv2.resize(frame_cam, (cam_panel_w, DISPLAY_H))   # 9:16

        add_label(ref_panel, "Dance Reference", (0, 220, 220))
        add_label(you_panel, "You",              (0, 220,  80))

        if paused:
            cv2.putText(ref_panel, "PAUSED", (panel_w//2-65, DISPLAY_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,80,255), 3)

        divider = np.full((DISPLAY_H, 4, 3), 80, dtype=np.uint8)
        combined = np.hstack([ref_panel, divider, you_panel])

        draw_hud(combined, smoothed, instant,
                 rhythm.score(), rhythm.beat_flash, tempo_bpm,
                 session.coach_lines, session.coach_until)

        cv2.imshow("Dance Mirror", combined)

        # ── Keys ─────────────────────────────────────────────────────────────
        # ── Frame rate limiter (30 fps, same as pose model) ───────────────────
        now = time.time()
        sleep_ms = max(1, int((next_frame - now) * 1000))
        next_frame += frame_dur

        key = cv2.waitKey(sleep_ms) & 0xFF
        if   key == ord('q'): break
        elif key == ord('r'):
            cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            score_hist.clear()
            rhythm.reset(beat_times)
            video_start = None
            paused = False
        elif key == ord('p'):
            paused = not paused
        elif key == ord('c'):
            request_coaching(session, rhythm, on_coach_done)

    # ── Auto-fire coach on exit ─────────────────────────────────────────────
    request_coaching(session, rhythm,
                     lambda lines: print("\nCoach:\n" + "\n".join(lines)))

    det_vid.close()
    det_cam.close()
    cap_vid.release()
    cap_cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python app.py <video.mp4 or YouTube URL>")
    main(sys.argv[1])
