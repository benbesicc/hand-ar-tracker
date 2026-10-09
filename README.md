# hand ar tracker

*Real-time hand AR system with ASCII art effects, holographic ribbon, and gesture controls built with Python + MediaPipe.*

---

# table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Gesture Controls](#gesture-controls)
7. [Keyboard Controls](#keyboard-controls)
8. [Effect Modes](#effect-modes)
9. [Configuration](#configuration)
10. [Project Architecture](#project-architecture)
11. [License](#license)

---

# overview

Hand AR Tracker captures your webcam feed and overlays real-time augmented reality effects driven entirely by **hand gestures**, **spacebar presses**, and **eye winks** no buttons, no keyboard, just your body (plus the spacebar for toggling invert mode).

The core effect is a **holographic ASCII ribbon** that stretches between your two hands, rendering your silhouette as scrolling ASCII scanlines against a black background. Pressing **spacebar** inverts the entire effect, and a **double wink** locks the ribbon's position in 3D space.

---

# features

- **Dual-hand tracking** - up to two hands simultaneously via MediaPipe Hands
- **Holographic ASCII ribbon** - scanline particle effect renders your silhouette as ASCII art between both hands
- **Space-to-reverse** - press the spacebar to flip the effect: real video inside ribbon, ASCII art outside
- **Double-wink lock** - wink twice in one second to freeze the ribbon position in space
- **Person + object segmentation** - MediaPipe Selfie Segmentation + background subtraction
- **Live FPS counter** with rolling average
- **Click-to-flip** camera rotation button in the HUD
- Auto GPU/CPU selection (CUDA or CPU)
- Fully configurable via `config.yaml`

---

# hardware requirements

# minimum

| Component | Requirement |
|-----------|-------------|
| Camera    | Any USB/built-in webcam |
| Python    | 3.12.7 |
| RAM       | 4 GB |
| OS        | Windows 10+, macOS 12+, Linux |

# recommended

| Component | Recommendation |
|-----------|----------------|
| GPU       | NVIDIA GPU with CUDA |
| Camera    | 1080p webcam at 60 fps |
| RAM       | 8 GB+ |
| CPU       | AMD Ryzen 5 / Intel i5 or better |

---

# installation

# windows (one-click)

1. **Clone the repo**
   ```bash
   git clone https://github.com/benbesicc/hand-ar-tracker.git
   cd hand-ar-tracker
   ```

2. **Run the installer** - creates the virtual environment and installs all dependencies:
   ```
   install.bat
   ```

3. **Launch the tracker:**
   ```
   start.bat
   ```

# manual installation

```bash
git clone https://github.com/benbesicc/hand-ar-tracker.git
cd hand-ar-tracker

python -m venv venv

# windows
venv\Scripts\activate
# macos  linux
source venv/bin/activate

pip install -r requirements.txt

python main.py
```

---

# quick start

```bash
start.bat        # windows one-click launch
# or
python main.py   # manual
```

The app opens your webcam. Show **both hands** to the camera to start.

---

# tutorial stepbystep walkthrough

1. **Launch** – run `start.bat` (Windows) or `python main.py`.
2. **Calibration** – the system automatically detects your hands and face. Wait a couple of seconds for the FPS counter to stabilize.
3. **Enable the ribbon** – perform a **double‑pinch** (thumb and index together on each hand, then touch the pinched tips). The holographic ASCII ribbon appears between your hands.
4. **Reverse mode** – press **spacebar**. The ribbon now shows the *real* video inside while the rest of the scene turns ASCII.
5. **Lock the ribbon** – wink one eye twice within one second. The ribbon’s position freezes in space, allowing you to move your hands without moving the effect.
6. **Unlock** – repeat the double‑wink to release the lock.
7. **Toggle off** – repeat the double‑pinch gesture to turn the ribbon off.
8. **Adjust settings** – edit `config.yaml` to change camera resolution, confidence thresholds, colors, or to enable extra gesture labels.
9. **Exit** – press **Q** or **ESC** (hand‑skeleton toggle) to close the application.

---

# controls summary

| Control | Action |
|---|---|
| **Spacebar** | Switch to reverse mode (real video inside ribbon, ASCII outside) |
| **Q** | Quit application |
| **ESC** | Toggle hand skeleton visibility |
| **Delete** | Show/hide HUD (FPS, flip button, device label) |

---

---

# gesture controls

Most effects are controlled entirely by **body gestures** - no keyboard required for tracking or locking the AR effects.

# double pinch touch - toggle ribbon onoff

Bring both hands in front of the camera, **pinch** your thumb and index finger together on **each hand simultaneously**, then touch the two pinched fingertips together.

- **First double-pinch:** Ribbon turns **ON** (ASCII scanline effect between hands)
- **Second double-pinch:** Ribbon turns **OFF**
- If either hand leaves frame while the ribbon is off, the effect auto-resets

```
Both hands visible → pinch both → touch pinched tips together → toggle
```

# double wink - lock  unlock ribbon position

Wink **one eye** (not both - a full blink won't count) **twice within 1 second** to lock the ribbon at its current position in space. The ribbon will stay frozen even if you move your hands away.

- **Double-wink while ribbon is ON:** Position locks. You can lower your hands - the ribbon stays.
- **Double-wink while ribbon is locked:** Unlocks the ribbon so it follows your hands again.

---

# keyboard controls

| Key | Action |
|-----|--------|
| **Spacebar** | Switch to reverse mode (real video inside ribbon, ASCII outside) |
| **Q** | Quit the application |
| **ESC** | Toggle hand skeleton visibility (dots + lines) on/off |
| **Delete** | Toggle the entire HUD (FPS counter, flip button, device label) on/off |

# hud - on-screen controls

| Element | Location | Action |
|---------|----------|--------|
| **FPS counter** | Top-left | Live rolling-average frame rate |
| **Flip 180 button** | Top-center (clickable) | Click to rotate camera 180° for upside-down mounts |
| **Device label** | Top-right | Shows `CUDA` or `CPU` |

---

# effect modes

# default mode (no reverse toggle)

```
Ribbon OFF:  Normal camera feed
Ribbon ON:   Your silhouette rendered as scrolling ASCII scanlines
             between your two hands, against a black background.
             Rest of frame is plain black.
```

# reversed mode (after spacebar press)

```
Ribbon OFF:  Full frame converted to ASCII art on black background
Ribbon ON:   Inside ribbon = clear real video
             Outside ribbon = ASCII art
             (No border - clean hard edge)
```

# locked ribbon (after double wink)

The ribbon polygon stays fixed in screen-space at the position captured when the wink fired. Works in both default and reversed modes.

---

# configuration

All settings live in `config.yaml`.

# camera

```yaml
camera:
  index: 0          # webcam index (0  default)
  width: 1280
  height: 720
  fps: 60
  rotate_180: false  # true if camera is mounted upside-down
```

# tracking

```yaml
tracking:
  max_hands: 2
  min_detection_confidence: 0.8
  min_tracking_confidence: 0.7
  model_complexity: 1   # 0  litefast 1  fullaccurate
```

# renderer

```yaml
renderer:
  skeleton_color: [0, 255, 0]     # bgr - green
  keypoint_color: [0, 0, 255]     # bgr - red
  keypoint_radius: 6
  skeleton_thickness: 2
  show_fps: true
  show_coordinates: false
  show_gesture_label: false
  show_hand_label: false
  show_device_label: true
```

# gestures

```yaml
gestures:
  enabled: false         # enable named gesture labels (pinch fist etc)
  pinch_threshold: 0.05  # thumb-to-index distance to count as a pinch
  fist_threshold: 0.85
```

# device

```yaml
device: auto   # auto  cuda0  cpu
```

---

# project architecture

| File | Purpose |
|------|---------|
| `main.py` | Entry point, main loop, wink detection, spacebar toggle, mode routing |
| `tracker.py` | MediaPipe Hands wrapper + FaceMesh + SelfieSegmentation |
| `renderer.py` | All OpenCV drawing: skeleton, ribbon (ASCII + real), HUD |
| `ascii_processor.py` | Vectorised ASCII art renderer (CLAHE + char tiles) |
| `gesture.py` | Per-hand gesture classification |
| `config.yaml` | All runtime configuration |
| `utils/fps_counter.py` | Rolling-window FPS counter |
| `utils/device_utils.py` | GPU/CPU auto-detection |
| `utils/math_utils.py` | Landmark smoothing and coordinate helpers |
| `install.bat` | One-click Windows dependency installer |
| `start.bat` | One-click Windows launcher |
| `assets/` | Reference images and ASCII character data |

---

# license

MIT License - see [LICENSE](LICENSE) for details.
