import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageOps

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=280, height=280, bg='white', highlightthickness=1, highlightbackground="black")
        self.canvas.pack(pady=10)

        self.predict_button = Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10)

        self.prediction_label = Label(root, text="Prediction: ", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        try:
            self.model = load_model('mnist_model.h5')
            print("Model loaded successfully.")
        except (IOError, ImportError):
            print("Error: mnist_model.h5 not found. Please train the model first.")
            self.root.destroy()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        self.prediction_label.config(text=f"Prediction: {predicted_digit}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Prediction: ")

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
