import requests
import json
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import cv2

url = "http://127.0.0.1:5001/predict"

font_paths = [
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]
font_p = next((p for p in font_paths if os.path.exists(p)), None)

# Helper function to generate clean white-on-black handwritten/font text images
def generate_text_image(text, size=(800, 300), font_size=64):
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    if font_p:
        font = ImageFont.truetype(font_p, font_size)
        # Center the text
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        else:
            w, h = 400, font_size
        x = (size[0] - w) // 2
        y = (size[1] - h) // 2
        draw.text((x, y), text, fill=255, font=font)
    else:
        draw.text((20, 20), text, fill=255)
        
    # Apply standard 2x2 dilation for natural handwriting stroke
    arr = np.array(img)
    kernel = np.ones((2, 2), np.uint8)
    arr = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(arr)

# Test cases definition
test_cases = {
    "Digits": {
        "text": "3 7 0",
        "size": (300, 100),
        "font_size": 48,
        "mode": "auto"
    },
    "Alphabets": {
        "text": "A B C",
        "size": (300, 100),
        "font_size": 48,
        "mode": "auto"
    },
    "Words": {
        "text": "Handwritten",
        "size": (500, 150),
        "font_size": 56,
        "mode": "auto"
    },
    "Sentences": {
        "text": "The quick brown fox jumps over the lazy dog.",
        "size": (800, 150),
        "font_size": 36,
        "mode": "auto"
    },
    "Paragraphs": {
        "text": "Deep learning models are highly effective.\nThey recognize handwritten pages accurately.",
        "size": (800, 300),
        "font_size": 32,
        "mode": "auto"
    },
    "Page": {
        "text": "Handwritten Text Recognition Project.\n\nThis capstone final year project uses Microsoft TrOCR.\nIt segments paragraphs, lines, and words.\n\nThen it extracts editable digital text with confidence.",
        "size": (900, 500),
        "font_size": 28,
        "mode": "auto"
    }
}

print("======================================================")
print("RUNNING AUTOMATED HTR PIPELINE INTEGRATION TESTS")
print("======================================================")

success_count = 0

for name, config in test_cases.items():
    print(f"\n[Test Case] {name}...")
    img = generate_text_image(config["text"], size=config["size"], font_size=config["font_size"])
    
    # Save to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Send request
    payload = {
        "image": f"data:image/png;base64,{img_str}",
        "source": f"Automated Pipeline Test - {name}",
        "model_mode": config["mode"]
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            res = response.json()
            print(f"  Input Type: {res['input_type']}")
            print(f"  Extracted Text:\n{res['text']}")
            print(f"  Confidence: {res['confidence']*100:.2f}%")
            print(f"  Prediction Time: {res['prediction_time']}")
            print(f"  Segments Found: {len(res['segments'])}")
            
            # Simple validation check: make sure we did not return "No text recognized" or empty
            if res['text'] and res['text'] != "No text recognized":
                print("  Status: PASS")
                success_count += 1
            else:
                print("  Status: FAIL (Empty text/No text recognized)")
        else:
            print(f"  Status: FAIL. HTTP status code: {response.status_code}")
    except Exception as e:
        print(f"  Status: EXCEPTION. Error: {e}")

print("\n======================================================")
print(f"HTR Pipeline Testing Complete: {success_count}/{len(test_cases)} Passed")
print("======================================================")
