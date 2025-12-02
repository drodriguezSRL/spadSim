#!/usr/bin/en python3

"""spad_emulator.py

Simulate binary SPAD frames from an input RGB video using optical-flow interpolation
and a Poisson photon detection model.

Usage (example):
    python spad_emulator.py input.mp4 --output_dir output_spad --rgb_fps 30 

Dependencies:
    - numpy
    - opencv-python (cv2)
    - pillow (PIL)
    - tqdm

The script will:
  1. Extract RGB frames from the input video at `rgb_fps` (optional to keep them).
  2. For each consecutive RGB image pair, compute Farneback optical flow and
     generate N_spad = round(spad_rate / rgb_fps) SPAD binary frames by
     motion-compensated warping + linear blending at intermediate time fractions.
  3. Convert blended intensity to expected photon counts using P_rgb and QE,
     add optional dark counts, sample Poisson counts and threshold to binary.
  4. Save all binary SPAD frames as single-channel PNGs (0 or 255) into a single
     folder, and save metadata.json and diagnostics.json.

To-implement next:
[x] implement export RGB frames
[] save_rgb flag 
[] crop RGB or SPAD to the right size already
[] if no rgb_fps input, instead of 30 used video's recorded fps 
[x] implement optical flow 
[x] implement warping function to interpolate spad frames for each rgb frame based on optical flow
[x] simulate SPAD frames...
[x] diagnostics and metadata
[] write README
[] improve photon flux estimation
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any 

# CONSTANTS AND DEFAULT PARAMETERS
EPSILON = 1e-9 # tolerance for floating point number comparisons
DEFAULT_FPS = 30.0 # fallback FPS if video reports 0 or fails
FILENAME_PAD = 6 # number of digits in frame filenames
SPAD_FPS = 100 # default SPAD FPS
SPAD_QE = 0.5 # default SPAD quantum efficiency
PHOTONS_PER_PX = 1000 # number of photons per pixel per RGB exposure at normalized intensity = 1 
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

def load_png_gray(p: str) -> np.ndarray:
    """
    Returns the grayscale (0-255) Numpy array of a PNG image
    """
    img = Image.open(p).convert('L') # 'L' luminance, 8bit grayscale
    return np.array(img, dtype=np.float32) # converts from PIL image object to numpy array. float32 is needed for optical flow.

def compute_farneback(
        img0_gray: np.ndarray, 
        img1_gray: np.ndarray
        ) -> np.ndarray:
    """
    Computes Farneback Optical Flow between two images (img0 and img1)
    Pixel movement convention: 
    - dx > 0: pixel moved right
    - dx < 0: pixel moved left
    - dy > 0: pixel moved down
    - dy < 0: pixel moved up
    
    Parameters:
    - img0_gray (np.ndarray): first grayscale image as float32 [0-255]
    - img1_gray (np.ndarray): second grayscale image as float32 [0-255]

    Returns:
    - A (H,W,2) array with (dx,dy) motion per pixel 
    """
    # convert to uint8 for opencv 
    i0 = np.clip(img0_gray, 0, 255).astype(np.uint8)
    i1 = np.clip(img1_gray, 0, 255).astype(np.uint8)

    # compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev=i0, # first image 
        next=i1, # second image
        flow=None, # ???
        pyr_scale=0.5, # image pyramid scale
        levels=3, # number of pyramid levels
        winsize=15, # window size for averaging motion
        iterations=3, # refinement iterations per level
        poly_n=5, poly_sigma=1.2, # polynomial expansion parameters 
        flags=0 # default behavior
    )

    return flow.astype(np.float32)

def warp_image(
        img_gray: np.ndarray, 
        flow: np.ndarray, 
        alpha: float
        ) -> np.ndarray:
    """
    Motion-aware image interpolation. Computes a warped version of the original image shiften along the motion vectors by a fraction 'alpha' of the total movement.
    - alpha = 0 -> outputs = original frame
    - alpha = 1 -> output = next frame according to the flow field
    - alpha = 0.5 -> halfwaybetween the two frames (motion-interpolated)

    Parameters:
    - img_gray (np.ndarray): grayscale image to be warped as float32 [0-255]
    - flow (np.ndarrays): optical flow matrix (field) HxWx2
    - alpha (float): fraction of pixel warp

    Returns:
    - A (H,W,2) array with (dx,dy) motion per pixel 
    """
    # get image size
    h, w = img_gray.shape[:2]
    
    # create a grid of pixel coordinates ("address" of each pixel)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # compute displacements
    dx = (alpha * flow[..., 0]).astype(np.float32) # flow[..., 0] = horizontal pixel movement (dx)
    dy = (alpha * flow[..., 1]).astype(np.float32) # flow[..., 1] = vertical pixel movement (dy)

    # compute sampling positions with forward warping via inverse mapping
    map_x= (xs + dx).astype(np.float32)
    map_y= (ys + dy).astype(np.float32)

    # remap the image via bilinear interpolation based on the new positions defined by (map_x, map_y). Note: bilinear interpolation handles fractional pixel positions; if coordinates fall outside the image, border pixels are repeated (REPLICATE)
    warped = cv2.remap(img_gray.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return warped

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

def generate_spad_sequence_from_pair(
        imgA_gray: np.ndarray,
        imgB_gray: np.ndarray,
        flow: np.ndarray,
        n_spad_per_pair: int,
        rgb_photons: float, 
        QE: float,
        t_spad: float, 
        t_rgb: float,
        include_dcr: bool, 
        dcr: float,
        detection_threshold: int,
        output_dir: str, 
        global_index_start: int,
        rng: np.random.Generator
        ) -> Tuple[int, Dict[str, Any]]:
    """
    Generate SPAD frames for a single RGB pair (A->B) and save them to disk. 
    For this, the function interpolates intermediate frames between A and B using optical flow (flow matrix obtained using compute_farneback()) and transforms them into binary frames simulating the photon emission and detection of a SPAD camera.

    Parameters:
    - img*_gray (np.ndarrays): the two grayscale images as nparrays HxW (0-255 floats)
    - flow (np.ndarrays): optical flow matrix (field) HxWx2
    - n_spad_per_pair (int): approximated number of SPAD frames pair RGB pair
    - rgb_photons (float): number of photons per pixel (total photon flux) for one RGB frame when the normalized pixel intensity is maximum (i = 1)
    - QE (float): quantum efficiency of the SPAD camera
    - t_spad (float): exposure time of the SPAD camera (in sec)
    - t_rgb (float): exposure time of the RGB camera (in sec)
    - include_dcr (bool): whether to include dark counts 
    - dcr (float): dark counts per seconds per pixel
    - detection_threshold (int): minimum number of photons that must be detected for a bright pixel (>=1)
    - output_dir (str): output directory for saving the generated SPAD frames
    - global_index_start (int): starting global numbering for all SPAD frames
    - rng (np.random.Generator): a numpy random number generator

    Returns:
    - A tuple containing the next global index (int) and the diagnostics (Dict[str, Any]) for this pair.
    """
    # store image dimensions
    h, w = imgA_gray.shape[:2]

    # diagnostics
    diagnostics = {
        'pair_frames_generated': 0,
        'mean_lambda_signal': 0.0,
        'mean_lambda_dark': 0.0,
        'mean_lambda_total': 0.0,
        'mean_detection_prob_empirical': 0.0
    }

    N = n_spad_per_pair
    if N <=0:
        return global_index_start, diagnostics
    
    # initialize accumulators for SPAD frame stats
    sum_lambda_signal = 0.0
    sum_lambda_dark = 0.0
    sum_lambda_total = 0.0
    sum_detected = 0.0
    total_pixels = float(h*w*N)
    
    # initialize index (used for naming output frames)
    idx = global_index_start

    # main loop for generation of 'N' frames between imgA and imgB
    for j in range(N):
        alpha = j / max(1,N) # fraction of movement in [0,1)

        # motion compensated warp 
        A_warp = warp_image(imgA_gray, flow, alpha)
        B_warp = warp_image(imgB_gray, flow, alpha - 1.0) # warp B backwards 

        # blend both warped A and B frames
        blended = (1.0 - alpha) * A_warp + alpha * B_warp

        # normalize intensity values of the blended frame
        i_norm = np.clip(blended / 255.0, 0.0, 1.0)

        # compute the EXPECTED photon arrival rate (lambda_signal)
        # this assumes photon arrival follows a poissonian distribution
        lambda_signal = (rgb_photons * i_norm * QE * (t_spad/t_rgb)).astype(np.float64)

        # dark noise component (if enabled)
        lambda_dark = (dcr * t_spad) if include_dcr else 0.0

        # total mean per pixel
        lambda_total = lambda_signal + lambda_dark

        # simulate ACTUAL photon arrivals per pixel using Poisson noise
        n_samples = rng.poisson(lam=lambda_total)

        # threshold to get binary detection
        bits = (n_samples >= detection_threshold).astype(np.uint8) # 0 or 1

        # accumulate statistics
        sum_lambda_signal += float(lambda_signal.mean())
        sum_lambda_dark += float(lambda_dark)  # same for all pixels if constant dark_rate
        sum_lambda_total += float(lambda_total.mean())
        sum_detected += bits.mean()

        # save SPAD frame to disk
        img_out = (bits*255).astype(np.uint8)
        img = Image.fromarray(img_out).convert("1")  # convert to true 1-bit mode
        out_path = Path(output_dir) / f"spad_{idx:07d}.png"
        img.save(out_path)

        idx +=1

    # update diagnostics
    diagnostics['pair_frames_generated'] = N
    diagnostics['mean_lambda_signal'] = sum_lambda_signal / max(1, N)
    diagnostics['mean_lambda_dark'] = sum_lambda_dark / max(1, N)
    diagnostics['mean_lambda_total'] = sum_lambda_total / max(1, N)
    diagnostics['mean_detection_prob_empirical'] = sum_detected / max(1, N)

    return idx, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Simulate binary SPAD frames from an RGB video.")
    parser.add_argument("input_video", type=str, help="Path to the input RGB video file.")
    parser.add_argument("--output_dir", "-o", type=str, default="/output_dir", help="Path to the output directory for frames and metadata.")
    parser.add_argument("--rgb_fps", "-f", type=float, default=DEFAULT_FPS, help="Target FPS for RGB frame extraction (Hz).")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Maximum number of RGB frames to extract (for testing).")
    parser.add_argument("--spad_rate", "-sf", type=float, default=SPAD_FPS, help="SPAD frame rate (Hz)")
    parser.add_argument("--rgb_photons", "-p", type=float, default=PHOTONS_PER_PX, help="Number of photons per pixel for one RGB frame when the normalized pixel intensity is maximum (i = 1).")
    parser.add_argument("--quantum_efficiency", "-qe", type=float, default=SPAD_QE, help="Quantum efficiency (0..1)")
    parser.add_argument("--include_dcr", "-id", type=int, default=INCLUDE_DCR, choices=[True,False], help="Include dark counts (True/False)")
    parser.add_argument("--dcr", "-d", type=float, default=SPAD_DCR, help="Dark count rate per pixel (counts/s)")
    parser.add_argument("--detection_threshold", "-dt", type=int, default=DETECTION_THRESHOLD, help="Detection threshold (int >=1)")
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

    # Extract RGB frames from input video
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

    # Generate random number of photon arrivals per pixel
    if args.seed == 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.seed)

    # Prepare metadata
    metadata = {
        'input_video': args.input_video,
        'rgb_fps': rgb_fps,
        'spad_rate': spad_fps,
        't_rgb': t_rgb,
        't_spad': t_spad,
        'n_spad_per_pair': n_spad_per_pair,
        'rgb_photons': args.rgb_photons,
        'QE': args.quantum_efficiency,
        'include_dark_counts': bool(args.include_dcr),
        'dark_rate': args.dcr,
        'detection_threshold': args.detection_threshold,
        'optical_flow_method': args.optical_flow_method,
        'save_rgb_frames': bool(args.save_rgb),
        'seed': int(args.seed)
    } 

    # Generate SPAD frames from RGB pairs
    global_idx = 0
    all_diagnostics = []
    print('📸 Processing RGB pairs and generating SPAD frames...')
    for k in tqdm(range(num_frames - 1), desc='Processing RGB pairs'):
        # get two consecutive RGB frames
        pathA = rgb_paths[k]
        pathB = rgb_paths[k + 1]

        # load them in grayscale
        grayA = load_png_gray(pathA)
        grayB = load_png_gray(pathB)

        # compute flow from A to B
        flow = compute_farneback(grayA, grayB)

        # generate spad frames for this pair
        next_idx, diag = generate_spad_sequence_from_pair(
            imgA_gray=grayA, 
            imgB_gray=grayB, 
            flow=flow,
            n_spad_per_pair=n_spad_per_pair,
            rgb_photons=args.rgb_photons, 
            QE=args.quantum_efficiency,
            t_spad=t_spad, 
            t_rgb=t_rgb,
            include_dcr=bool(args.include_dcr), 
            dcr=args.dcr,
            detection_threshold=args.detection_threshold,
            output_dir=spad_dir, 
            global_index_start=global_idx,
            rng=rng
        )

        global_idx = next_idx
        all_diagnostics.append({'pair_index': k, **diag})

    # finalize metadata 
    metadata['total_spad_frames'] = global_idx
    metadata_path = Path(out_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # finalize diagnostics
    diagnostics_path = Path(out_dir) / 'diagnostics.json'
    summary = {
        'pairs_processed': len(all_diagnostics),
        'total_spad_frames': global_idx,
        'per_pair_stats': all_diagnostics
    }
    # compute aggregate stats
    if len(all_diagnostics) > 0:
        mean_lambda_signal = float(np.mean([d['mean_lambda_signal'] for d in all_diagnostics]))
        mean_lambda_dark = float(np.mean([d['mean_lambda_dark'] for d in all_diagnostics]))
        mean_lambda_total = float(np.mean([d['mean_lambda_total'] for d in all_diagnostics]))
        mean_detection_prob = float(np.mean([d['mean_detection_prob_empirical'] for d in all_diagnostics]))
    else:
        mean_lambda_signal = mean_lambda_dark = mean_lambda_total = mean_detection_prob = 0.0
    summary['aggregate'] = {
        'mean_lambda_signal': mean_lambda_signal,
        'mean_lambda_dark': mean_lambda_dark,
        'mean_lambda_total': mean_lambda_total,
        'mean_detection_prob_empirical': mean_detection_prob
    }
    with open(diagnostics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('✅ Done')
    print(f'📥 Metadata saved to: {metadata_path}')
    print(f'📉 Diagnostics saved to: {diagnostics_path}')
    print(f'🔥 Total SPAD frames written: {global_idx} (to directory: {spad_dir})')

if __name__ == "__main__":
    main()
