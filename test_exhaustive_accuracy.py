import os
import json
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO

# Configuration
SERVER_URL = "http://127.0.0.1:5001/predict"

def generate_augmented_image(text, size=(800, 300), font_size=40, font_name="Arial", 
                             slant=0.0, rotate_angle=0.0, blur_radius=0.0, 
                             noise_level=0.0, gradient_bg=False, thickness="normal"):
    """
    Generates a high-fidelity synthetic handwriting image with various augmentations
    representing actual difficult cases (cursive slant, rotation, noise, blur, lighting).
    """
    # Background color: 0 is black, 255 is white. We will create white text on a black background
    # since that matches drawing canvas inputs.
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    
    # Load Font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    # Measure text size
    # Check compatibility for older PIL versions
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    else:
        text_w, text_h = draw.textsize(text, font=font)
        
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2
    
    # Render Text
    if thickness == "thick":
        # Draw multiple times with minor offsets to simulate thick strokes
        for offset_x in [-2, -1, 0, 1, 2]:
            for offset_y in [-2, -1, 0, 1, 2]:
                draw.text((x + offset_x, y + offset_y), text, fill=255, font=font)
    elif thickness == "thin":
        # Draw normally
        draw.text((x, y), text, fill=200, font=font) # slightly lower brightness
    else:
        draw.text((x, y), text, fill=255, font=font)
        
    # 1. Slant (Cursive Simulation)
    if slant != 0.0:
        # Affine transform for horizontal shear
        img = img.transform(size, Image.AFFINE, (1, slant, 0, 0, 1, 0), resample=Image.BICUBIC)
        
    # 2. Rotation
    if rotate_angle != 0.0:
        img = img.rotate(rotate_angle, resample=Image.BICUBIC, expand=False)
        
    # 3. Blur
    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        
    # 4. Salt and Pepper Noise
    if noise_level > 0.0:
        arr = np.array(img)
        # Salt
        num_salt = np.ceil(noise_level * arr.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
        arr[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(noise_level * arr.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
        arr[tuple(coords)] = 0
        img = Image.fromarray(arr)
        
    # 5. Lighting Gradients (Camera photo simulation)
    if gradient_bg:
        arr = np.array(img).astype(np.float32)
        gradient = np.linspace(0, 50, size[0], dtype=np.float32)
        gradient_2d = np.tile(gradient, (size[1], 1))
        arr = np.clip(arr + gradient_2d, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
    return img

def query_prediction(img_pil):
    """
    Sends the PIL image to the Flask /predict endpoint.
    """
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    payload = {
        "image": f"data:image/png;base64,{img_str}"
    }
    
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error querying predict API: {e}")
    return None

def normalize_text(text):
    return text.strip().lower().replace(" ", "").replace(".", "").replace(",", "").replace("-", "")

def calculate_accuracy(expected, predicted):
    e = normalize_text(expected)
    p = normalize_text(predicted)
    if not e and not p:
        return 100.0
    if not e or not p:
        return 0.0
    matches = sum(1 for c in e if c in p)
    return (matches / max(len(e), len(p))) * 100.0

# Define Benchmark Batches
digits_batch = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
alphabets_batch = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]
mixed_batch = ["ABC123", "HELLO123", "MCA2026", "AI123", "TEST001"]
words_batch = [
    "handwritten", "project", "university", "console", "dashboard",
    "database", "accuracy", "prediction", "neural", "network",
    "learning", "academic", "demonstration", "science", "recognition"
]
sentences_batch = [
    "Deep learning models are highly effective.",
    "This final year project uses Microsoft TrOCR.",
    "The quick brown fox jumps over the lazy dog.",
    "It segments paragraphs, lines, and words."
]
paragraphs_batch = [
    "Deep learning models are highly effective.\nThey recognize handwritten pages accurately.",
    "Handwritten Text Recognition Project.\nThis capstone final year project uses Microsoft TROCR.\nIt segments paragraphs, lines, and words."
]

print("======================================================")
print("RUNNING EXHAUSTIVE ACCURACY AND METRIC BENCHMARK")
print("======================================================")

results_summary = {
    "digits": {"executed": 0, "passed": 0, "sum_acc": 0.0},
    "alphabets": {"executed": 0, "passed": 0, "sum_acc": 0.0},
    "mixed": {"executed": 0, "passed": 0, "sum_acc": 0.0},
    "words": {"executed": 0, "passed": 0, "sum_acc": 0.0},
    "sentences": {"executed": 0, "passed": 0, "sum_acc": 0.0},
    "paragraphs": {"executed": 0, "passed": 0, "sum_acc": 0.0}
}

total_tests = 0
passed_tests = 0
sum_confidence = 0.0
sum_time = 0.0

# 1. Run Digits Batch (10 variations per digit = 100 samples)
print("\n[Benchmarking] Digits (100 samples with variations)...")
for d in digits_batch:
    variations = [
        {"slant": -0.1, "rotate": 5.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": 0.1, "rotate": -5.0, "blur": 0.0, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.5, "noise": 0.01, "thick": "thin"},
        {"slant": 0.2, "rotate": 8.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": -0.2, "rotate": -8.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": 0.0, "rotate": 10.0, "blur": 0.8, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": -10.0, "blur": 0.0, "noise": 0.02, "thick": "thin"},
        {"slant": 0.15, "rotate": 3.0, "blur": 0.3, "noise": 0.00, "thick": "normal"},
        {"slant": -0.15, "rotate": -3.0, "blur": 0.0, "noise": 0.01, "thick": "normal"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.0, "noise": 0.00, "thick": "normal"}
    ]
    for idx, var in enumerate(variations):
        img = generate_augmented_image(d, size=(120, 120), font_size=48, 
                                       slant=var["slant"], rotate_angle=var["rotate"], 
                                       blur_radius=var["blur"], noise_level=var["noise"],
                                       thickness=var["thick"])
        res = query_prediction(img)
        if res:
            pred = res.get("text", "")
            conf = res.get("confidence", 0.0)
            p_time_str = res.get("prediction_time", "0.0s")
            p_time = float(p_time_str.replace("s", ""))
            
            acc = calculate_accuracy(d, pred)
            results_summary["digits"]["executed"] += 1
            results_summary["digits"]["sum_acc"] += acc
            sum_confidence += conf
            sum_time += p_time
            total_tests += 1
            
            if acc >= 80.0:
                results_summary["digits"]["passed"] += 1
                passed_tests += 1

# 2. Run Alphabets Batch (5 variations per letter = 260 samples)
print("[Benchmarking] Alphabets (260 samples with variations)...")
for char in alphabets_batch:
    variations = [
        {"slant": 0.1, "rotate": 5.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": -0.1, "rotate": -5.0, "blur": 0.0, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.5, "noise": 0.01, "thick": "thin"},
        {"slant": 0.15, "rotate": 4.0, "blur": 0.2, "noise": 0.00, "thick": "normal"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.0, "noise": 0.00, "thick": "normal"}
    ]
    for var in variations:
        img = generate_augmented_image(char, size=(120, 120), font_size=48, 
                                       slant=var["slant"], rotate_angle=var["rotate"], 
                                       blur_radius=var["blur"], noise_level=var["noise"],
                                       thickness=var["thick"])
        res = query_prediction(img)
        if res:
            pred = res.get("text", "")
            conf = res.get("confidence", 0.0)
            p_time_str = res.get("prediction_time", "0.0s")
            p_time = float(p_time_str.replace("s", ""))
            
            acc = calculate_accuracy(char, pred)
            results_summary["alphabets"]["executed"] += 1
            results_summary["alphabets"]["sum_acc"] += acc
            sum_confidence += conf
            sum_time += p_time
            total_tests += 1
            
            if acc >= 80.0:
                results_summary["alphabets"]["passed"] += 1
                passed_tests += 1

# 3. Run Mixed Alphanumeric Batch (5 variations = 25 samples)
print("[Benchmarking] Mixed Alphanumeric (25 samples)...")
for mix in mixed_batch:
    variations = [
        {"slant": 0.1, "rotate": 3.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": -0.1, "rotate": -3.0, "blur": 0.0, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.4, "noise": 0.01, "thick": "thin"},
        {"slant": 0.0, "rotate": 5.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.0, "noise": 0.00, "thick": "normal"}
    ]
    for var in variations:
        img = generate_augmented_image(mix, size=(300, 100), font_size=36, 
                                       slant=var["slant"], rotate_angle=var["rotate"], 
                                       blur_radius=var["blur"], noise_level=var["noise"],
                                       thickness=var["thick"])
        res = query_prediction(img)
        if res:
            pred = res.get("text", "")
            conf = res.get("confidence", 0.0)
            p_time_str = res.get("prediction_time", "0.0s")
            p_time = float(p_time_str.replace("s", ""))
            
            acc = calculate_accuracy(mix, pred)
            results_summary["mixed"]["executed"] += 1
            results_summary["mixed"]["sum_acc"] += acc
            sum_confidence += conf
            sum_time += p_time
            total_tests += 1
            
            if acc >= 80.0:
                results_summary["mixed"]["passed"] += 1
                passed_tests += 1

# 4. Run Words Batch (5 variations = 75 samples)
print("[Benchmarking] Words (75 samples)...")
for w in words_batch:
    variations = [
        {"slant": 0.1, "rotate": 3.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": -0.1, "rotate": -3.0, "blur": 0.0, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.4, "noise": 0.01, "thick": "thin"},
        {"slant": 0.0, "rotate": 5.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.0, "noise": 0.00, "thick": "normal"}
    ]
    for var in variations:
        img = generate_augmented_image(w, size=(400, 120), font_size=32, 
                                       slant=var["slant"], rotate_angle=var["rotate"], 
                                       blur_radius=var["blur"], noise_level=var["noise"],
                                       thickness=var["thick"])
        res = query_prediction(img)
        if res:
            pred = res.get("text", "")
            conf = res.get("confidence", 0.0)
            p_time_str = res.get("prediction_time", "0.0s")
            p_time = float(p_time_str.replace("s", ""))
            
            acc = calculate_accuracy(w, pred)
            results_summary["words"]["executed"] += 1
            results_summary["words"]["sum_acc"] += acc
            sum_confidence += conf
            sum_time += p_time
            total_tests += 1
            
            if acc >= 80.0:
                results_summary["words"]["passed"] += 1
                passed_tests += 1

# 5. Run Sentences Batch (5 variations = 20 samples)
print("[Benchmarking] Sentences (20 samples)...")
for s in sentences_batch:
    variations = [
        {"slant": 0.05, "rotate": 2.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": -0.05, "rotate": -2.0, "blur": 0.0, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.3, "noise": 0.01, "thick": "thin"},
        {"slant": 0.0, "rotate": 4.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.0, "noise": 0.00, "thick": "normal"}
    ]
    for var in variations:
        img = generate_augmented_image(s, size=(800, 150), font_size=28, 
                                       slant=var["slant"], rotate_angle=var["rotate"], 
                                       blur_radius=var["blur"], noise_level=var["noise"],
                                       thickness=var["thick"])
        res = query_prediction(img)
        if res:
            pred = res.get("text", "")
            conf = res.get("confidence", 0.0)
            p_time_str = res.get("prediction_time", "0.0s")
            p_time = float(p_time_str.replace("s", ""))
            
            acc = calculate_accuracy(s, pred)
            results_summary["sentences"]["executed"] += 1
            results_summary["sentences"]["sum_acc"] += acc
            sum_confidence += conf
            sum_time += p_time
            total_tests += 1
            
            if acc >= 75.0:
                results_summary["sentences"]["passed"] += 1
                passed_tests += 1

# 6. Run Paragraphs Batch (5 variations = 10 samples)
print("[Benchmarking] Paragraphs (10 samples)...")
for p in paragraphs_batch:
    variations = [
        {"slant": 0.05, "rotate": 1.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": -0.05, "rotate": -1.0, "blur": 0.0, "noise": 0.00, "thick": "thick"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.2, "noise": 0.01, "thick": "thin"},
        {"slant": 0.0, "rotate": 3.0, "blur": 0.0, "noise": 0.00, "thick": "normal"},
        {"slant": 0.0, "rotate": 0.0, "blur": 0.0, "noise": 0.00, "thick": "normal"}
    ]
    for var in variations:
        img = generate_augmented_image(p, size=(800, 300), font_size=24, 
                                       slant=var["slant"], rotate_angle=var["rotate"], 
                                       blur_radius=var["blur"], noise_level=var["noise"],
                                       thickness=var["thick"])
        res = query_prediction(img)
        if res:
            pred = res.get("text", "")
            conf = res.get("confidence", 0.0)
            p_time_str = res.get("prediction_time", "0.0s")
            p_time = float(p_time_str.replace("s", ""))
            
            acc = calculate_accuracy(p, pred)
            results_summary["paragraphs"]["executed"] += 1
            results_summary["paragraphs"]["sum_acc"] += acc
            sum_confidence += conf
            sum_time += p_time
            total_tests += 1
            
            if acc >= 70.0:
                results_summary["paragraphs"]["passed"] += 1
                passed_tests += 1

# Output Final Stats Table
print("\n======================================================")
print("ACCURACY BENCHMARK SUMMARY")
print("======================================================")
print(f"Total test samples generated & evaluated: {total_tests}")
print(f"Total passed predictions (acc >= threshold): {passed_tests}")
print(f"Total failed predictions: {total_tests - passed_tests}")
print(f"Overall model accuracy rating: {(passed_tests / total_tests) * 100.0:.2f}%")
print(f"Average confidence index: {(sum_confidence / total_tests) * 100.0:.2f}%")
print(f"Average processing time per sample: {sum_time / total_tests:.4f}s")
print("------------------------------------------------------")
for cat, stats in results_summary.items():
    avg_acc = (stats["sum_acc"] / stats["executed"]) if stats["executed"] > 0 else 0.0
    pass_rate = (stats["passed"] / stats["executed"]) * 100.0 if stats["executed"] > 0 else 0.0
    print(f"Category: {cat.upper():<12} | Count: {stats['executed']:<4} | Pass Rate: {pass_rate:.1f}% | Avg Similarity Acc: {avg_acc:.2f}%")
print("======================================================")
