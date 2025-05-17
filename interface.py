import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

class FODDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FOD Detection Interface")
        
        # Load the YOLO model
        self.model = YOLO('fod50.pt')
        
        # Initialize variables
        self.current_image = None
        self.original_image = None  # Add this line
        self.photo = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Create buttons
        self.load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(btn_frame, text="Reset", command=self.reset, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display label
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            # Load and display the image
            self.current_image = cv2.imread(file_path)
            self.original_image = self.current_image.copy()  # Store original image
            self.display_image(self.current_image)
            self.predict_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
    
    def predict(self):
        if self.current_image is not None:
            # Run inference without confidence threshold
            results = self.model(self.current_image)
            # Get annotated frame
            annotated_frame = results[0].plot()
            # Display the result
            self.display_image(annotated_frame)
    
    def reset(self):
        # Clear the image label
        self.image_label.config(image='')
        self.image_label.image = None
        
        # Reset all image variables
        self.current_image = None
        self.original_image = None
        self.photo = None
        
        # Disable buttons
        self.predict_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
    
    def display_image(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image if too large (optional)
        max_size = 800
        height, width = image_rgb.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        image_pil = Image.fromarray(image_rgb)
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

if __name__ == "__main__":
    root = tk.Tk()
    app = FODDetectionApp(root)
    root.mainloop()
