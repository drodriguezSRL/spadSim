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

The SPAD simulator models photon arrivals using poissonian statistics and simulates the ultra-fast frame rates of a SPAD by interpolating motion between the extracted RGB frames using optical flow.

A SPAD camera operates differently from conventioanl CMOS/CCD imaging sensors. SPADs record binary frames (1-bit output) based on the detection of a single photon per pixel (0: no photon, 1: photon) at ultra-fast speeds (up to 100kfps, µs-exposure per frame). For more information about SPADs and how they compare to conventional cameras, I encourage you to read [this paper](https://arxiv.org/abs/2510.10597). 

This simulator emulates SPAD imaging by:

1. Extracting RGB frames from the input video 
2. Estimating the photon flux in the image  
3. Interpolating motion between RGB frames using optical flow  
4. Simulating Poisson photon arrivals  
5. Thresholding detections to generate a sequence of binary frames  

## Imaging Model

### 1. Computing photon flux from RGB 8-bit pixel intensity

One of the first things we need to model is the photon flux; i.e., the total number of photons arriving to each pixel at any given time. Ideally, we should know the total number of photons emitted by a scene per second. This way the photon flux could be calcualted by simply dividing this number by the pixel active surface area. However, estimating total photon emissions in a scene is non-trivial. Instead, I have simplified this estimation via photon-count scaling by first normalizing the **intensity values of each pixel** (`I(x,y)`) in the RGB frames:

```math
i(x,y) = I(x,y) / 255
```

And then defining a high-level, user-friendly parameter, `rgb_photons`, that represents how many signal photons are collected by a single pixel during a full RGB exposure when than pixel has `i=1` (i.e., `I=255` = white/saturated pixel). This way the user can control and define the overall brightness of the scene. 

![WARNING]
> This is a simplification that's dependent on the light sensitivity of the RGB camera used to record the input video. Scene features that aren't captured by the RGB camera (e.g., clipped shadows and highlights) won't show up in the binary frames, even if, in reality, a SPAD camera may be capable of resolving those same features due to its enhanced sensitivity. 

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

