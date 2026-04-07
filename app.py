"""
Dance Mirror — real-time dance coaching with pose comparison and AI feedback.
Display framework mirrors the flappy bird project (pygame + clock.tick).

Usage:
    python app.py <video.mp4>
    python app.py <YouTube URL>

Controls:
    q / ESC  quit
    r        restart video
    p        pause / resume
    c        request AI coach feedback
"""

import sys, os, time, threading, subprocess, queue
from collections import deque, defaultdict

import cv2
import numpy as np
import pygame
import mediapipe as mp

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import ollama as _ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    import yt_dlp
    YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

# ── MediaPipe Tasks ────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
OLLAMA_MODEL   = "llama3.2"

BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

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

DETECT_W = 640   # downscale before detection for speed

class AsyncDetector:
    """
    Runs pose detection in a background thread.
    The render loop submits frames and reads the latest result without blocking.
    """
    def __init__(self, detector):
        self.detector  = detector
        self.in_q      = queue.Queue(maxsize=1)
        self._lm       = None
        self._vec      = None
        self._coords   = None
        self._lock     = threading.Lock()
        self._ts       = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            rgb = self.in_q.get()
            self._ts += 33
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect_for_video(mp_img, self._ts)
            lm = result.pose_landmarks[0] if result.pose_landmarks else None
            vec, coords = normalise(lm)
            with self._lock:
                self._lm, self._vec, self._coords = lm, vec, coords

    def submit(self, bgr):
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (DETECT_W, int(h * DETECT_W / w)))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        try:
            self.in_q.put_nowait(rgb)
        except queue.Full:
            pass   # detector is busy — skip this frame, use last result

    def result(self):
        with self._lock:
            return self._lm, self._vec, self._coords

# ── Landmark indices ───────────────────────────────────────────────────────────
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HI, R_HI = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

KEY_LM = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HI, R_HI, L_KN, R_KN, L_AN, R_AN]

LM_NAMES = {
    L_SH: "Left shoulder",  R_SH: "Right shoulder",
    L_EL: "Left elbow",     R_EL: "Right elbow",
    L_WR: "Left wrist",     R_WR: "Right wrist",
    L_HI: "Left hip",       R_HI: "Right hip",
    L_KN: "Left knee",      R_KN: "Right knee",
    L_AN: "Left ankle",     R_AN: "Right ankle",
}

BODY_CONNECTIONS = [
    (L_SH, R_SH),
    (L_SH, L_EL), (L_EL, L_WR),
    (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HI), (R_SH, R_HI),
    (L_HI, R_HI),
    (L_HI, L_KN), (L_KN, L_AN),
    (R_HI, R_KN), (R_KN, R_AN),
]

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FPS       = 30
DISPLAY_H        = 480
SCORE_SMOOTH     = 20
AUDIO_COOLDOWN   = 3.5
POSE_THRESHOLD   = 0.30
VELOCITY_THRESH  = 0.06
BEAT_WINDOW      = 0.18
COACH_DISPLAY_SEC = 12

# ── Audio cue ──────────────────────────────────────────────────────────────────
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
    if landmarks is None:
        return None, None
    def xy(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y], dtype=np.float32)
    hip   = (xy(L_HI) + xy(R_HI)) / 2.0
    scale = np.linalg.norm((xy(L_SH) + xy(R_SH)) / 2.0 - hip) + 1e-6
    coords, flat = {}, []
    for idx in KEY_LM:
        n = (xy(idx) - hip) / scale
        coords[idx] = n
        flat.extend(n)
    return np.array(flat, dtype=np.float32), coords

def mirror_vec(v: np.ndarray) -> np.ndarray:
    m = v.copy()
    for i in range(0, len(KEY_LM) - 1, 2):
        lx, ly = i*2, i*2+1
        rx, ry = (i+1)*2, (i+1)*2+1
        m[lx], m[rx] = -v[rx], -v[lx]
        m[ly], m[ry] =  v[ry],  v[ly]
    return m

def cosine_pct(a, b) -> float:
    if a is None or b is None: return 0.0
    d = np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
    return float(np.clip((np.dot(a, b) / d + 1) / 2 * 100, 0, 100))

def best_pose_score(ref, you) -> float:
    if ref is None or you is None: return 0.0
    return max(cosine_pct(ref, you), cosine_pct(mirror_vec(ref), you))

# ── Beat detection ─────────────────────────────────────────────────────────────

def load_beats(video_path: str):
    if not LIBROSA_OK:
        return [], 0
    try:
        print("Analysing audio ...", end=" ", flush=True)
        y, sr = librosa.load(video_path, mono=True, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        print(f"done  ({bpm:.1f} BPM)")
        return beat_times, bpm
    except Exception as e:
        print(f"failed ({e})")
        return [], 0

# ── Rhythm tracker ─────────────────────────────────────────────────────────────

class RhythmTracker:
    def __init__(self, beat_times):
        self.beat_times  = np.array(beat_times)
        self.next_beat_i = 0
        self.hits        = []
        self.prev_coords = None

    def update(self, current_sec, coords_you):
        velocity = 0.0
        if self.prev_coords and coords_you:
            for idx in KEY_LM:
                if idx in coords_you and idx in self.prev_coords:
                    velocity += float(np.linalg.norm(coords_you[idx] - self.prev_coords[idx]))
        self.prev_coords = coords_you
        if len(self.beat_times) and self.next_beat_i < len(self.beat_times):
            if current_sec > self.beat_times[self.next_beat_i] + BEAT_WINDOW:
                self.hits.append(velocity > VELOCITY_THRESH)
                self.next_beat_i += 1

    def score(self) -> float:
        return sum(self.hits) / len(self.hits) * 100 if self.hits else 0.0

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

    def update(self, instant, coords_ref, coords_you):
        self.pose_scores.append(instant)
        if coords_ref and coords_you:
            for idx in KEY_LM:
                if idx in coords_ref and idx in coords_you:
                    self.joint_errors[idx].append(
                        float(np.linalg.norm(coords_you[idx] - coords_ref[idx])))

    def avg_score(self): return float(np.mean(self.pose_scores)) if self.pose_scores else 0.0
    def worst_joints(self, n=3):
        avgs = {lm: float(np.mean(v)) for lm, v in self.joint_errors.items() if v}
        return sorted(avgs.items(), key=lambda x: x[1], reverse=True)[:n]
    def duration(self): return time.time() - self.start

# ── AI coach ───────────────────────────────────────────────────────────────────

def request_coaching(session, rhythm, on_done):
    if not OLLAMA_OK:
        on_done(["AI coach unavailable — pip install ollama"]); return
    if session.coach_pending: return
    session.coach_pending = True

    def _call():
        try:
            worst = session.worst_joints(3)
            worst_str = ", ".join(f"{LM_NAMES.get(lm)} ({e:.2f})" for lm, e in worst) or "none"
            prompt = (f"You are a dance coach. Stats: pose {session.avg_score():.1f}%, "
                      f"rhythm {rhythm.score():.1f}%, worst joints: {worst_str}, "
                      f"duration {session.duration():.0f}s. "
                      f"Give 3 short coaching tips, one sentence each, plain text, newline-separated.")
            resp  = _ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}])
            lines = [l.strip() for l in resp["message"]["content"].strip().split("\n") if l.strip()][:3]
        except Exception as e:
            lines = [f"Coach error: {e}"]
        session.coach_pending = False
        on_done(lines)

    threading.Thread(target=_call, daemon=True).start()

# ── Skeleton drawing (OpenCV, on BGR frames) ───────────────────────────────────

def _jcolor(dist: float):
    t = min(dist / (POSE_THRESHOLD * 2), 1.0)
    return (0, int(200*(1-t)), int(220*t))

def draw_skeleton(frame, landmarks, coords_you=None, coords_ref=None):
    if landmarks is None: return
    h, w = frame.shape[:2]
    def pt(idx): return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    for a, b in BODY_CONNECTIONS:
        if landmarks[a].visibility < 0.3 or landmarks[b].visibility < 0.3: continue
        if coords_you and coords_ref:
            da = float(np.linalg.norm(coords_you.get(a,np.zeros(2)) - coords_ref.get(a,np.zeros(2))))
            db = float(np.linalg.norm(coords_you.get(b,np.zeros(2)) - coords_ref.get(b,np.zeros(2))))
            color = tuple(int(x) for x in (np.array(_jcolor(da)) + np.array(_jcolor(db))) // 2)
            thick = 6
        else:
            color, thick = (200, 200, 0), 4
        cv2.line(frame, pt(a), pt(b), color, thick)

    for idx in KEY_LM:
        if landmarks[idx].visibility < 0.3: continue
        if coords_you and coords_ref:
            dist = float(np.linalg.norm(
                coords_you.get(idx, np.zeros(2)) - coords_ref.get(idx, np.zeros(2))))
            color, radius = _jcolor(dist), 14
        else:
            color, radius = (0, 220, 220), 10
        cv2.circle(frame, pt(idx), radius, color, -1)
        cv2.circle(frame, pt(idx), radius, (255, 255, 255), 2)

# ── Pygame helpers ─────────────────────────────────────────────────────────────

def bgr_to_surface(bgr, w, h):
    """Convert a BGR OpenCV frame to a pygame Surface at size (w, h)."""
    frame = cv2.resize(bgr, (w, h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.rot90(frame))

def draw_label(screen, font, text, x, y, color):
    shadow = font.render(text, True, (0, 0, 0))
    label  = font.render(text, True, color)
    screen.blit(shadow, (x+2, y+2))
    screen.blit(label,  (x,   y))

def draw_coach_overlay(screen, font_big, font_sm, lines, coach_until, total_w, total_h):
    if not lines or time.time() >= coach_until:
        return
    pad, lh = 14, 28
    box_h = pad*2 + lh*(len(lines)+1)
    box_x, box_y = 40, (total_h - box_h) // 2 - 20
    box_w = total_w - 80
    overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    overlay.fill((10, 10, 10, 200))
    screen.blit(overlay, (box_x, box_y))
    title = font_big.render("AI Coach", True, (0, 200, 255))
    screen.blit(title, (box_x + pad, box_y + pad))
    for i, line in enumerate(lines):
        txt = font_sm.render(line, True, (240, 240, 240))
        screen.blit(txt, (box_x + pad, box_y + pad + lh*(i+2)))

# ── Audio pose cues ────────────────────────────────────────────────────────────

def check_pose_cues(coords_ref, coords_you):
    if not coords_ref or not coords_you: return
    def dy(idx): return float(coords_ref[idx][1] - coords_you[idx][1])
    def dx(idx): return float(coords_ref[idx][0] - coords_you[idx][0])
    T = POSE_THRESHOLD
    for cond, msg, key in [
        (dy(L_WR) >  T, "Left arm higher!",     "la_up"),
        (dy(L_WR) < -T, "Left arm lower!",      "la_dn"),
        (dy(R_WR) >  T, "Right arm higher!",    "ra_up"),
        (dy(R_WR) < -T, "Right arm lower!",     "ra_dn"),
        (dx(L_WR) >  T, "Extend left arm!",     "la_out"),
        (dx(L_WR) < -T, "Bring left arm in!",   "la_in"),
        (dx(R_WR) >  T, "Bring right arm in!",  "ra_in"),
        (dx(R_WR) < -T, "Extend right arm!",    "ra_out"),
        (dy(L_KN) >  T, "Bend left knee!",      "lk"),
        (dy(R_KN) >  T, "Bend right knee!",     "rk"),
    ]:
        if cond: speak(msg, key); break

# ── Video download / cache ─────────────────────────────────────────────────────

def resolve_video(source: str) -> str:
    if not source.startswith("http"):
        return source
    if not YTDLP_OK:
        sys.exit("[ERROR] pip install yt-dlp")
    import hashlib
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(source.encode()).hexdigest()[:10]
    cached   = os.path.join(cache_dir, f"{url_hash}.mp4")
    if os.path.exists(cached):
        print(f"Using cached video: {cached}"); return cached
    out_tmpl = os.path.join(cache_dir, f"{url_hash}.%(ext)s")
    opts = {"format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": out_tmpl, "quiet": True, "merge_output_format": "mp4"}
    print(f"Downloading: {source}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
        path = ydl.prepare_filename(info)
        for ext in (".webm", ".mkv"): path = path.replace(ext, ".mp4")
    if not os.path.exists(path):
        for f in os.listdir(cache_dir):
            if url_hash in f: path = os.path.join(cache_dir, f); break
    print(f"Saved to: {path}"); return path

# ── Main ───────────────────────────────────────────────────────────────────────

def main(source: str):
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[ERROR] Model not found: {MODEL_PATH}")

    video_path = resolve_video(source)

    cap_vid = cv2.VideoCapture(video_path)
    cap_cam = cv2.VideoCapture(0)
    if not cap_vid.isOpened(): sys.exit(f"[ERROR] Cannot open: {video_path}")
    if not cap_cam.isOpened(): sys.exit("[ERROR] Cannot open webcam")

    video_fps = cap_vid.get(cv2.CAP_PROP_FPS) or 30.0
    video_w   = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h   = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_panel_w = int(video_w * DISPLAY_H / video_h)
    cam_panel_w = int(DISPLAY_H * 9 / 16)
    total_w     = ref_panel_w + 4 + cam_panel_w

    beat_times, _ = load_beats(video_path)
    rhythm  = RhythmTracker(beat_times)
    session = SessionTracker()
    score_hist = deque(maxlen=SCORE_SMOOTH)

    print("Creating pose detectors ...")
    det_vid = AsyncDetector(make_detector())
    det_cam = AsyncDetector(make_detector())
    print("Ready.\n")

    # ── Pygame init (same pattern as flappy bird) ──────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((total_w, DISPLAY_H))
    pygame.display.set_caption("Dance Mirror")
    clock  = pygame.time.Clock()
    font_big = pygame.font.Font(None, 36)
    font_sm  = pygame.font.Font(None, 26)

    paused      = False
    frame_vid   = None
    video_start = None

    def on_coach_done(lines):
        session.coach_lines = lines
        session.coach_until = time.time() + COACH_DISPLAY_SEC
        speak(". ".join(lines), "coach")

    print("Dance Mirror —  q/ESC: quit   r: restart   p: pause   c: AI coach\n")

    running = True
    while running:
        # ── Events ─────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    score_hist.clear()
                    rhythm.reset(beat_times)
                    video_start = None
                    paused = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_c:
                    request_coaching(session, rhythm, on_coach_done)

        # ── Webcam ─────────────────────────────────────────────────────────────
        ret_w, raw = cap_cam.read()
        if not ret_w: break
        frame_cam = cv2.flip(raw, 1)

        # ── Dance video (natural playback + frame-skip to stay in sync) ────────
        if not paused:
            if video_start is None:
                video_start = time.time()
            elapsed     = time.time() - video_start
            video_pos   = cap_vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # Skip frames when we fall behind — never seek (too expensive)
            while elapsed > video_pos + 1.0 / video_fps:
                ret_v, frame_vid = cap_vid.read()
                if not ret_v: break
                video_pos = cap_vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ret_v, frame_vid = cap_vid.read()
            if not ret_v:
                cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                score_hist.clear(); rhythm.reset(beat_times); video_start = None
                continue

        current_sec = cap_vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ── Pose detection (async — non-blocking) ───────────────────────────────
        det_vid.submit(frame_vid)
        det_cam.submit(frame_cam)
        lm_vid, vec_ref, coords_ref = det_vid.result()
        lm_cam, vec_you, coords_you = det_cam.result()

        instant = best_pose_score(vec_ref, vec_you)
        score_hist.append(instant)
        session.update(instant, coords_ref, coords_you)

        if not paused:
            rhythm.update(current_sec, coords_you)

        check_pose_cues(coords_ref, coords_you)

        # ── Draw skeletons onto frames (OpenCV) ─────────────────────────────────
        draw_skeleton(frame_vid, lm_vid)
        draw_skeleton(frame_cam, lm_cam, coords_you, coords_ref)

        # ── Convert frames → pygame surfaces ───────────────────────────────────
        screen.fill((20, 20, 20))
        ref_surf = bgr_to_surface(frame_vid, ref_panel_w, DISPLAY_H)
        cam_surf = bgr_to_surface(frame_cam, cam_panel_w, DISPLAY_H)
        screen.blit(ref_surf, (0, 0))
        screen.blit(cam_surf, (ref_panel_w + 4, 0))

        # ── Labels ─────────────────────────────────────────────────────────────
        draw_label(screen, font_big, "Dance Reference", 12, 10, (0, 220, 220))
        draw_label(screen, font_big, "You", ref_panel_w + 16, 10, (0, 220, 80))

        if paused:
            txt = font_big.render("PAUSED", True, (0, 80, 255))
            screen.blit(txt, (ref_panel_w // 2 - txt.get_width() // 2, DISPLAY_H // 2))

        # ── AI coach overlay ────────────────────────────────────────────────────
        draw_coach_overlay(screen, font_big, font_sm,
                           session.coach_lines, session.coach_until,
                           total_w, DISPLAY_H)

        pygame.display.flip()
        clock.tick(TARGET_FPS)   # same as flappy bird

    # ── Cleanup ────────────────────────────────────────────────────────────────
    request_coaching(session, rhythm,
                     lambda lines: print("\nCoach:\n" + "\n".join(lines)))
    cap_vid.release(); cap_cam.release()
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python app.py <video.mp4 or YouTube URL>")
    main(sys.argv[1])
