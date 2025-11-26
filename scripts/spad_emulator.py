#!/usr/bin/en python3

"""
Implement next:
- save_rgb flag 
- crop RGB or SPAD to the right size already 
- simulate SPAD frames...
"""

import os
import cv2
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple 

# CONSTANTS AND DEFAULT PARAMETERS
EPSILON = 1e-9 # tolerance for floating point number comparisons
DEFAULT_FPS = 30.0 # fallback FPS if video reports 0 or fails
FILENAME_PAD = 6 # number of digits in frame filenames
SPAD_FPS = 100 # default SPAD FPS
SPAD_QE = 0.5 # default SPAD quantum efficiency
PHOTONS_PER_PX = 30 # number of photons per pixel per RGB exposure at normalized intensity = 1 
INCLUDE_DCR = False # include SPAD dark count rate in the model
SPAD_DCR = 100.0 # default dark count rate per pixel (counts/sec)
OPTFLOW_METHOD = "farneback" # default optical flow method
DETECTION_THRESHOLD = 1 # detection threshold
SAVE_RGB = True
BITPACK = False
DIAGNOSTIC = True # compile a diagnostic report 
SEED = 0 # random seed (0 = random)

# UTILS
def ensure_dir(p: str):
    """
    Ensure that a directory exists; create it if it doesn't.
    """
    Path(p).mkdir(parents=True, exist_ok=True)


# CORE FUNCTIONALITY
def extract_frames_from_video(
        video_path: str, 
        out_dir: str,
        target_fps: float,
        max_frames: Optional[int] = None
        ) -> Tuple[int,list]:
    """
    Extract frames from a video at a target FPS and save them as PNG images.

    Parameters:
    - video_path (str): Path to the input video file.
    - out_dir (str): Directory to save the extracted frames.
    - target_fps (float): Desired frames per second for extraction.
    - max_frames (Optional[int]): Optional maximum number of frames to extract.

    Returns:
    - A tuple containing the number of extracted frames and a list of their file paths.
    """
    # open video file
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file {video_path}")
    except:
        cap.release()
        raise

    # get video metadata
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # compute frame extraction interval
    step = orig_fps / target_fps

    frame_idx = 1
    out_paths = []
    extracted = 0

    # progress bar
    total = total_frames if max_frames is None else min(total_frames, max_frames)    
    pbar = tqdm(total=total, desc="Extracting frames", unit='frame')

    # read and extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx / max(1, step) - extracted >= 1 - EPSILON:
            # save frame
            fname = f"frame_{extracted:0{FILENAME_PAD}d}.png"
            out_path = Path(out_dir) / fname # os.path.join(out_dir, fname)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # openCV uses BGR by default
            Image.fromarray(rgb).save(out_path) # directly saving with cv2.imwrite(str(out_path), frame) may be faster for high-volume extraction [to check]
            out_paths.append(out_path)
            extracted += 1

            # stop early if max_frames reached
            if max_frames is not None and extracted >= max_frames:
                break

            frame_idx += 1
            pbar.update(1)
    
    # cleanup
    pbar.close()
    cap.release()

    # returns number of extracted frames and their paths
    return extracted, out_paths 


def main():
    parser = argparse.ArgumentParser(description="Simulate binary SPAD frames from an RGB video.")
    parser.add_argument("input_video", type=str, help="Path to the input RGB video file.")
    parser.add_argument("--output_dir", "-o", type=str, default="/output_dir", help="Path to the output directory for frames and metadata.")
    parser.add_argument("--rgb_fps", "-rf", type=float, default=DEFAULT_FPS, help="Target FPS for RGB frame extraction (Hz).")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Maximum number of RGB frames to extract (for testing).")
    parser.add_argument("--spad_rate", "-sf", type=float, default=SPAD_FPS, help="SPAD frame rate (Hz)")
    parser.add_argument("--max_photons", "-p", type=float, default=PHOTONS_PER_PX, help="Number of photons per pixel for one RGB frame when the normalized pixel intensity is maximum (i = 1).")
    parser.add_argument("--quantum_efficiency", "-qe", type=float, default=SPAD_QE, help="Quantum efficiency (0..1)")
    parser.add_argument("--include_dcr", "-id", type=int, default=INCLUDE_DCR, choices=[True,False], help="Include dark counts (True/False)")
    parser.add_argument("--dcr", "-d", type=float, default=SPAD_DCR, help="Dark count rate per pixel (counts/s)")
    parser.add_argument("--optical_flow_method", '-ofm', type=str, default=OPTFLOW_METHOD, choices=['farneback'], help="Optical flow method")
    parser.add_argument("--save_rgb", "-s", type=bool, default=SAVE_RGB, choices=[True,False], help="Save extracted RGB frames (True/False)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (0 means random)")

    args = parser.parse_args()

    # prepare output directories
    out_dir = args.output_dir
    ensure_dir(out_dir)
    rgb_dir = Path(out_dir) / "rgb_frames" 
    ensure_dir(rgb_dir)
    spad_dir = Path(out_dir) / "spad_frames"
    ensure_dir(spad_dir)

    ## Extract RGB frames from input video
    print('⏳Extracting RGB frames from video...')
    num_frames, rgb_paths = extract_frames_from_video(
        video_path=args.input_video,
        out_dir=rgb_dir,
        target_fps= args.rgb_fps,
        max_frames = args.max_frames
    )
    if num_frames < 2:
        raise RuntimeError(f"At least 2 extracted RGB frames are required to generate interpolared SPAD frames.")
    print(f'✅ Extracted {num_frames} frames to {rgb_dir}')

    # Compute timing parameters
    rgb_fps = float(args.rgb_fps)
    spad_fps = float(args.spad_rate) 
    t_rgb = 1.0/rgb_fps 
    t_spad = 1.0/spad_fps

    n_spad_per_pair = int(round(t_rgb/t_spad)) # number of SPAD frames pair RGB pair (approx)
    if n_spad_per_pair < 1:
        n_spad_per_pair = 1
    print(f"[INFO] SPAD frames per RGB interval (approx): {n_spad_per_pair}")






if __name__ == "__main__":
    main()
