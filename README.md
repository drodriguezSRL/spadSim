# spadSim - Simulating binary SPAD frames from a standard video

## Quick setup

### 1. Clone repo
```
git clone https://github.com/drodriguezSRL/spadSim
```

### 2. Install dependencies
```
pip install numpy opencv-python pillow tqdm
```

### 3. Run a quick demo of the simulator
```
python ./scripts/spad_emulator.py ./testing/demo.mp4 --output_dir testing
```

### 4. Explore the results
Inside `testing/` you will find:

- `rgb_frames/` → extracted RGB video frames  
- `spad_frames/` → simulated binary SPAD frames  
- `metadata.json` → simulation parameters  
- `diagnostics.json` → photon/detection statistics  

### (Optional) Access the docstring
The code is fully documented. You can access the docstring with:
```
python -c "import scripts/spad_emulator; help(spad_emulator)"
```
or using `pydoc`:
```
python -m pydoc scripts/spad_emulator
```
---

## What the simulator does

This repository contains `spad_emulator.py`, a script that simulates the acquisition of **binary Single-Photon Avalanche Diode (SPAD)** frames using a standard RGB video as a reference input.

The SPAD simulator models photon arrivals using poissonian statistics and simulates the ultra-fast frame rates of a SPAD by interpolating motion between RGB frames using optical flow.

A SPAD camera behaves differently from conventioanl CMOS/CCD imaging sensors. SPADs record binary frames (1-bit output) based on the detection of single photons per pixel (0: no photon, 1: photon) at ultra-fast speeds (up to 100kfps, µs-exposure per frame). 

This simulator emulates SPAD imaging by:

1. Extracting RGB frames  
2. Estimating the photon flux in the image  
3. Interpolating motion between RGB frames using optical flow  
4. Simulating Poisson photon arrivals  
5. Thresholding detections to define binary SPAD frames  

## Scientific Model

### 1. Computing photon flux from RGB 8-bit pixel intensity

Normalized:
```
i(x,y) = I(x,y) / 255
```

`P_rgb` controls how many photons arrive during a full RGB exposure when intensity = 1.

## 2. Optical Flow Interpolation

Farnebäck optical flow estimates motion. Interpolated frames are generated at intermediate times.

\[
I(t) = (1-α)\,warp(A, αF) + α\,warp(B, (α-1)F)
\]

## 3. Photon Arrivals (Poisson)

\[
n \sim Poisson(λ_{total})
\]

Where:

\[
λ_{signal} = P_{rgb} \cdot i \cdot QE \cdot (t_{spad}/t_{rgb})
\]

\[
λ_{dark} = dark\_rate \cdot t_{spad}
\]

\[
λ_{total} = λ_{signal} + λ_{dark}
\]

## 4. Binary Detection

A SPAD pixel outputs:

```
1  if n ≥ 1
0  otherwise
```

---

# ⚙️ Parameters

| Parameter | Description | Default |
|----------|-------------|---------|
| `--input_video` | Path to input video | required |
| `--output_dir` | Output directory | required |
| `--rgb_fps` | Frame extraction rate | 30 |
| `--spad_rate` | SPAD frame rate | 100 |
| `--P_rgb` | Photons per RGB exposure at intensity=1 | 30 |
| `--QE` | Quantum efficiency | 0.5 |
| `--include_dark_counts` | Enable dark counts? | 0 |
| `--dark_rate` | Dark counts per second | 100 |
| `--detection_threshold` | Photon threshold | 1 |
| `--max_frames` | Limit RGB frames | None |
| `--seed` | RNG seed | 0 |

---

# ▶️ Example

```
python spad_simulator.py --input_video input.mp4 --output_dir spad_frames --spad_rate 100 --P_rgb 30
```

---

# 📊 Diagnostics

`diagnostics.json` includes:

- Mean signal photon rate  
- Mean dark count rate  
- Total λ  
- Detection probability  

---

# 🔮 Future Extensions

- RAFT optical flow  
- Dead time modeling  
- Afterpulsing  
- Fill-factor models  
- Bit-packed outputs  

---

# 🏁 Summary

This simulator converts ordinary RGB videos into realistic SPAD-style binary photon frames using optical flow and Poisson photon arrival modeling.

