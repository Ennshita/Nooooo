import os
import time
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
try:
    from pywuffs import ImageDecoderType, PixelFormat
    from pywuffs.aux import ImageDecoder, ImageDecoderConfig, ImageDecoderFlags
    PYWUFFS_AVAILABLE = True
except ImportError:
    PYWUFFS_AVAILABLE = False
import argparse
from pathlib import Path

class TimeProfiler:
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

def process_pil(file_path, device):
    with TimeProfiler() as t:
        im_pil = Image.open(file_path).convert("RGB")
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(device)
    return t.total

def process_opencv(file_path, device):
    with TimeProfiler() as t:
        im_cv = cv2.imread(file_path, cv2.IMREAD_COLOR)
        im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        im_cv = cv2.resize(im_cv, (640, 640), interpolation=cv2.INTER_LINEAR)
        im_data = torch.from_numpy(im_cv).permute(2, 0, 1).float() / 255.0
        im_data = im_data[None].to(device)
    return t.total

def process_pywuffs(file_path, device):
    with TimeProfiler() as t:
        try:
            # Configure PyWuffs decoder for PNG with BGR pixel format
            config = ImageDecoderConfig()
            config.enabled_decoders = [ImageDecoderType.PNG]
            config.pixel_format = PixelFormat.BGR
            decoder = ImageDecoder(config)
            # Decode image
            decoding_result = decoder.decode(file_path)
            im_np = decoding_result.pixbuf
            # Convert BGR to RGB
            im_np = im_np[:, :, ::-1]  # Reverse channels
            # Resize to 640x640
            im_np = cv2.resize(im_np, (640, 640), interpolation=cv2.INTER_LINEAR)
            # Convert to tensor: HWC to CHW and normalize to [0,1]
            im_data = torch.from_numpy(im_np).permute(2, 0, 1).float() / 255.0
            im_data = im_data[None].to(device)
        except Exception as e:
            print(f"PyWuffs failed for {file_path}: {e}. Skipping PyWuffs for this image.")
            return None
    return t.total

def benchmark_image_loading(folder_path, device="cpu"):
    image_files = [f for f in Path(folder_path).glob("*.png")]
    if not image_files:
        print(f"No PNG files found in {folder_path}")
        return

    pil_times = []
    opencv_times = []
    pywuffs_times = []

    print(f"Processing {len(image_files)} PNG images sequentially...")

    for i, img_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {img_path.name}")
        # PyWuffs
        if PYWUFFS_AVAILABLE:
            latency_pywuffs = process_pywuffs(str(img_path), device)
            pywuffs_times.append(latency_pywuffs if latency_pywuffs is not None else float('inf'))
        else:
            pywuffs_times.append(None)
        # PIL
        latency_pil = process_pil(str(img_path), device)
        pil_times.append(latency_pil)

        # OpenCV
        latency_opencv = process_opencv(str(img_path), device)
        opencv_times.append(latency_opencv)

        

    # Compute average latencies (excluding None or inf values for PyWuffs)
    avg_pil = sum(pil_times) / len(pil_times)
    avg_opencv = sum(opencv_times) / len(opencv_times)
    valid_pywuffs = [t for t in pywuffs_times if t is not None and t != float('inf')]
    avg_pywuffs = sum(valid_pywuffs) / len(valid_pywuffs) if valid_pywuffs else None

    # Print results
    print("\nBenchmark Results (average latency per image in seconds):")
    print(f"PIL: {avg_pil:.6f} seconds")
    print(f"OpenCV: {avg_opencv:.6f} seconds")
    if PYWUFFS_AVAILABLE and valid_pywuffs:
        print(f"PyWuffs: {avg_pywuffs:.6f} seconds")
    else:
        print("PyWuffs: Not available or failed (check pywuffs installation or API)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark image loading with PIL, OpenCV, and PyWuffs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to folder containing PNG images")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use (cpu or cuda:0)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a valid directory")
    else:
        benchmark_image_loading(args.input, args.device)