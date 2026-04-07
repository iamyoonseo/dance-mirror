# Dance Mirror

Real-time dance learning system using MediaPipe pose detection, beat/rhythm scoring, and AI coaching.

## Features

- **Side-by-side view** — dance reference video on the left, your webcam on the right
- **Skeleton overlay** — color-coded joints drawn on top of your body (green = matching, red = off)
- **Pose accuracy score** — real-time percentage match with smoothed rolling average
- **Beat & rhythm scoring** — detects music beats from the video audio, scores whether your movements hit the beat
- **Audio cues** — speaks corrections aloud ("Left arm higher!", "Bend your right knee!")
- **Mirror correction** — automatically handles left/right orientation differences
- **AI coach** — press `c` to get 3 personalized coaching tips from a local LLM (Ollama)
- **YouTube URL support** — paste any YouTube/Shorts URL directly

## Setup

```bash
# Install dependencies
pip install mediapipe opencv-python numpy yt-dlp librosa soundfile ollama

# Download pose model (one-time, ~7MB)
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'pose_landmarker_lite.task'
)
print('Model downloaded')
"

# Install and set up Ollama (for AI coach)
brew install ollama
ollama serve &
ollama pull llama3.2
```

## Usage

```bash
# Local video
python app.py my_dance_video.mp4

# YouTube URL
python app.py "https://youtube.com/shorts/..."
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Restart video |
| `p` | Pause / Resume |
| `c` | Request AI coach feedback |

## Requirements

- Python 3.9+
- macOS (audio cues use the built-in `say` command)
- Webcam
- [Ollama](https://ollama.com) for AI coach (optional)
