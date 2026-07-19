import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model
model = load_model("model.h5")

# Create main window
window = tk.Tk()
window.title("Handwritten Text Recognition System")
window.geometry("400x500")

# Canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="black")
canvas.pack(pady=20)

# Image for drawing (white digit on black)
image = Image.new("L", (280, 280), color=0)
draw = ImageDraw.Draw(image)

# Draw function
def draw_lines(event):
    x, y = event.x, event.y
    r = 8  # brush size
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

canvas.bind("<B1-Motion>", draw_lines)

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=0)
    result_label.config(text="Draw a digit")

# Predict function
def predict_digit():
    # Resize to 28x28
    img = image.resize((28, 28))
    img = np.array(img)

    # Normalize
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit}")

# Save prediction image for documentation
def save_prediction_images():
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0

    sample_images = x_test[:5]

    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()

# Save digit categories (0–9)
def save_digit_categories():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), _ = mnist.load_data()

    plt.figure(figsize=(10, 4))

    for digit in range(10):
        index = np.where(y_train == digit)[0][0]
        plt.subplot(2, 5, digit+1)
        plt.imshow(x_train[index], cmap='gray')
        plt.title(f"Digit_{digit}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("digit_categories.png")
    plt.show()

# Buttons
btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.pack(pady=10)

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.pack(pady=10)

btn_save_pred = tk.Button(
    window,
    text="Save Predictions Image",
    command=save_prediction_images
)
btn_save_pred.pack(pady=10)

btn_save_cat = tk.Button(
    window,
    text="Save Digit Categories",
    command=save_digit_categories
)
btn_save_cat.pack(pady=10)

# Result label
result_label = tk.Label(window, text="Draw a digit", font=("Arial", 16))
result_label.pack(pady=20)

# Run app
window.mainloop()
