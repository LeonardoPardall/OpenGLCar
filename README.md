# Computer Graphics Project — Interactive 3D Scene (2025/2026)

This repository contains an interactive 3D graphics application written in **Python** using **OpenGL (via PyOpenGL)**. The demo renders a controllable vehicle in a simple environment, featuring a scene graph, basic physics, and dynamic lighting (day/night cycle).

## Authors
- **Leonardo Pardal** (61836)
- **Afonso Henriques** (61826)
- **Pedro Carvalho** (61800)

---

## Demo
[Video demonstration](https://www.youtube.com/watch?v=cogYM4YylCk)

---

## Installation & Run

### Prerequisites
- Python 3.8 or newer
- GPU with OpenGL 3.3 support

### Install dependencies
From the project root run:

```bash
pip install -r requirements.txt
```

### Run
From the project root, run the main script. If your code is in the `cg_proj` folder (renamed from `src`):

```bash
cd cg_proj
python main.py
```

---

## Controls

| Key | Action |
| :--- | :--- |
| **Arrow keys (↑ ↓ ← →)** | Drive the car (accelerate, brake, turn) |
| **Space** | Cycle camera mode (Free → Follow → Driver) |
| **F** | Toggle garage gate |
| **K / L** | Toggle car doors (left / right) |
| **W / A / S / D** | Move free camera |
| **Q / E** | Move free camera up / down |
| **Mouse** | Look around (free camera) |
| **Z** | Toggle fullscreen |
| **ESC** | Quit |



