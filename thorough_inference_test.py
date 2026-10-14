import os
import glob
import time
import random
import json
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import matplotlib.pyplot as plt

from predict import load_model, predict_image, get_inference_transform

def run_thorough_testing():
    print("=====================================================================")
    print("           THOROUGH INFERENCE TEST & BENCHMARK SUITE                 ")
    print("=====================================================================")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, device = load_model("best_mrbeast_scam_model.pth", device=device)
    transform = get_inference_transform()
    print(f"Model: ConvNeXt-Tiny | Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")
    
    random.seed(42)
    
    # -------------------------------------------------------------
    # 1. CATEGORICAL EVALUATION
    # -------------------------------------------------------------
    categories = {
        "Real MrBeast Scam Screenshots (True Positives)": {
            "files": random.sample(glob.glob("*_image.jpg"), 25),
            "expected_scam": True
        },
        "User Discord Memes (test2.png & Discord Parodies)": {
            "files": ["test2.png"] + random.sample(glob.glob("assets/scam_memes/*.jpg"), min(15, len(glob.glob("assets/scam_memes/*.jpg")))),
            "expected_scam": False
        },
        "Legitimate MrBeast Logos & Jimmy Faces": {
            "files": random.sample(glob.glob("assets/mrbeast_logos/*"), min(12, len(glob.glob("assets/mrbeast_logos/*")))) + \
                     random.sample(glob.glob("assets/jimmy_faces/*"), min(12, len(glob.glob("assets/jimmy_faces/*")))),
            "expected_scam": False
        },
        "Adversarial Composites (Logos & Phones over UI)": {
            "files": random.sample(glob.glob("adversarial_composites/*.jpg"), 20),
            "expected_scam": False
        },
        "Normal X / Twitter & Elon Musk Posts": {
            "files": random.sample(glob.glob("assets/elon_x_screenshots/*.jpg"), 20),
            "expected_scam": False
        },
        "User Base Adversarial Set (Cats, Discord, UIs)": {
            "files": random.sample(glob.glob("adversarial/*"), 25),
            "expected_scam": False
        }
    }
    
    cat_summary = []
    
    print("--- 1. Testing Accuracy Across Diverse Categories ---")
    for cat_name, cat_data in categories.items():
        files = [f for f in cat_data["files"] if os.path.exists(f)]
        expected = cat_data["expected_scam"]
        
        probs = []
        correct = 0
        for f in files:
            res = predict_image(f, model, device, transform, threshold=0.5)
            probs.append(res["scam_probability"])
            if res["is_scam"] == expected:
                correct += 1
                
        acc = (correct / len(files)) * 100 if files else 0
        avg_prob = np.mean(probs) if probs else 0
        min_prob = np.min(probs) if probs else 0
        max_prob = np.max(probs) if probs else 0
        
        cat_summary.append({
            "category": cat_name,
            "count": len(files),
            "accuracy": acc,
            "avg_scam_prob": avg_prob,
            "prob_range": (min_prob, max_prob)
        })
        
        expected_str = "SCAM (1)" if expected else "LEGIT/MEME (0)"
        print(f"\nCategory: {cat_name}")
        print(f"  Target: {expected_str} | Evaluated: {len(files)} samples")
        print(f"  Accuracy: {acc:.1f}% ({correct}/{len(files)})")
        print(f"  Scam Probability: Mean={avg_prob*100:.2f}% | Range=[{min_prob*100:.2f}%, {max_prob*100:.2f}%]")

    # -------------------------------------------------------------
    # 2. DETAILED BREAKDOWN ON SPECIFIC NOTABLE IMAGES
    # -------------------------------------------------------------
    print("\n\n--- 2. Key Image Inspections ---")
    notable = [
        ("test2.png", "User's Discord meme with phone displaying MrBeast logo in front of fake exitscam monitor"),
        ("1517944412502753390_1517944411768487996_image.jpg", "Real Scam: Twitter post offering $5,600 wolzabet bonus"),
        ("1558910958397685860_1558910957562888212_image.jpg", "Real Scam: Micawin 'Withdrawal Success $5600.00' modal"),
        ("assets/logos/mrbeast_panther_hires.png", "Official high-res MrBeast panther logo"),
        ("assets/faces/jimmy_donaldson_2026.png", "Official Jimmy Donaldson face portrait"),
    ]
    
    for f, desc in notable:
        if os.path.exists(f):
            res = predict_image(f, model, device, transform)
            status = "🚨 SCAM" if res["is_scam"] else "✅ SAFE/MEME"
            print(f"• File: {f}")
            print(f"  Description: {desc}")
            print(f"  Classification: {status} (Probability: {res['scam_probability']*100:.2f}%, Confidence: {res['confidence']*100:.2f}%)\n")

    # -------------------------------------------------------------
    # 3. ROBUSTNESS & PERTURBATION TESTING (on test2.png and real scam)
    # -------------------------------------------------------------
    print("--- 3. Robustness & Perturbation Stress Testing ---")
    test_targets = [
        ("test2.png", False),
        ("1517944412502753390_1517944411768487996_image.jpg", True)
    ]
    
    perturbation_results = []
    
    for base_file, exp_scam in test_targets:
        if not os.path.exists(base_file):
            continue
        base_im = Image.open(base_file).convert("RGB")
        target_name = "Meme (test2.png)" if not exp_scam else "Real Scam Screenshot"
        print(f"\nPerturbations applied to: {target_name}")
        
        # Define stress conditions
        stresses = [
            ("Original (Clean)", base_im),
            ("Heavy Gaussian Blur (Radius=4)", base_im.filter(ImageFilter.GaussianBlur(radius=4))),
            ("Extreme Brightness (+60%)", ImageEnhance.Brightness(base_im).enhance(1.6)),
            ("Extreme Darkness (-60%)", ImageEnhance.Brightness(base_im).enhance(0.4)),
            ("High Contrast (+80%)", ImageEnhance.Contrast(base_im).enhance(1.8)),
            ("Low Contrast (-60%)", ImageEnhance.Contrast(base_im).enhance(0.4)),
            ("Slight Rotation (+15 deg)", base_im.rotate(15, expand=False)),
            ("Center Zoom (50% Crop)", base_im.crop((base_im.width*0.25, base_im.height*0.25, base_im.width*0.75, base_im.height*0.75))),
            ("Severe Downscale & Re-upscale (64x64 pixelation)", base_im.resize((64, 64)).resize(base_im.size))
        ]
        
        for stress_name, stressed_im in stresses:
            tensor = transform(stressed_im).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    prob = model.predict_proba(tensor).item()
            is_scam = prob >= 0.5
            passed = (is_scam == exp_scam)
            pass_str = "PASS" if passed else "FAIL"
            print(f"  [{pass_str}] {stress_name:45s} -> Prob: {prob*100:5.2f}% (Verdict: {'SCAM' if is_scam else 'SAFE'})")
            perturbation_results.append({
                "target": target_name,
                "stress": stress_name,
                "prob": prob,
                "passed": passed
            })

    # -------------------------------------------------------------
    # 4. GPU LATENCY & THROUGHPUT BENCHMARKING
    # -------------------------------------------------------------
    print("\n\n--- 4. GPU Latency & Throughput Benchmark on RTX 3080 ---")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                _ = model.predict_proba(dummy_input)
    torch.cuda.synchronize()
    
    # Single-image latency
    num_runs = 100
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                _ = model.predict_proba(dummy_input)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    
    print(f"Single-Image Inference Latency (Batch size = 1):")
    print(f"  Mean:   {mean_lat:.2f} ms")
    print(f"  Median: {p50_lat:.2f} ms")
    print(f"  p95:    {p95_lat:.2f} ms")
    print(f"  p99:    {p99_lat:.2f} ms")
    
    # Batch throughput
    batch_sizes = [1, 8, 16, 32, 64]
    print(f"\nBatch Throughput (Images / second):")
    for bs in batch_sizes:
        batch_input = torch.randn(bs, 3, 224, 224, device=device)
        # warmup
        for _ in range(5):
            with torch.no_grad():
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    _ = model.predict_proba(batch_input)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        repeats = 30
        for _ in range(repeats):
            with torch.no_grad():
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    _ = model.predict_proba(batch_input)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        throughput = (bs * repeats) / total_time
        print(f"  Batch size {bs:2d}: {throughput:7.1f} images/sec (Latency per batch: {(total_time/repeats)*1000:.2f} ms)")

    # -------------------------------------------------------------
    # 5. SCORE SEPARATION & CALIBRATION ANALYSIS
    # -------------------------------------------------------------
    print("\n--- 5. Score Separation & Margin Distribution ---")
    pos_samples = random.sample(glob.glob("*_image.jpg"), 100)
    neg_samples = ["test2.png"] + random.sample(glob.glob("adversarial_composites/*.jpg"), 50) + random.sample(glob.glob("assets/scam_memes/*.jpg"), min(49, len(glob.glob("assets/scam_memes/*.jpg"))))
    
    pos_scores = [predict_image(p, model, device, transform)["scam_probability"] for p in pos_samples if os.path.exists(p)]
    neg_scores = [predict_image(n, model, device, transform)["scam_probability"] for n in neg_samples if os.path.exists(n)]
    
    print(f"Positive (Scam) Scores (N={len(pos_scores)}):")
    print(f"  Min: {min(pos_scores)*100:.2f}% | Median: {np.median(pos_scores)*100:.2f}% | Max: {max(pos_scores)*100:.2f}%")
    print(f"Negative (Legit/Meme) Scores (N={len(neg_scores)}):")
    print(f"  Min: {min(neg_scores)*100:.2f}% | Median: {np.median(neg_scores)*100:.2f}% | Max: {max(neg_scores)*100:.2f}%")
    
    margin = min(pos_scores) - max(neg_scores)
    print(f"\nSeparation Margin (Lowest Scam Score - Highest Legit Score): {margin*100:+.2f}%")
    if margin > 0:
        print(">>> CLEAR ZERO-OVERLAP MARGIN: Decision boundary threshold 0.50 yields 100% classification separation!")
    else:
        print(f">>> Slight overlap near boundary (Separation difference: {margin*100:.2f}%)")
        
    print("\n=====================================================================")
    print("                    INFERENCE TESTING COMPLETE                       ")
    print("=====================================================================")

if __name__ == "__main__":
    run_thorough_testing()
