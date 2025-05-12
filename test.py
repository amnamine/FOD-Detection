import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np

class FODDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("FOD Detection System")
        
        # Initialize model
        self.model = YOLO('fod.pt')
        
        # Initialize variables
        self.image_path = None
        self.current_image = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create canvas for image display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(pady=10)
        
        # Create button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Create buttons
        self.load_btn = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = tk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(button_frame, text="Reset", command=self.reset)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if self.image_path:
            # Load and display image
            self.current_image = cv2.imread(self.image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_image)
    
    def predict(self):
        if self.image_path:
            # Perform prediction
            results = self.model.predict(source=self.image_path, conf=0.25)
            
            # Get the first result
            result = results[0]
            
            # Create a copy of the original image
            annotated_image = self.current_image.copy()
            
            # Draw boxes
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                conf = box.conf[0].numpy()
                cls = int(box.cls[0].numpy())
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Add label
                label = f'FOD: {conf:.2f}'
                cv2.putText(annotated_image, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Display annotated image
            self.display_image(annotated_image)
    
    def reset(self):
        # Clear image variables
        self.image_path = None
        self.current_image = None
        
        # Reset canvas to default size
        self.canvas.config(width=800, height=600)
        # Clear all items on canvas
        self.canvas.delete("all")
    
    def display_image(self, image):
        # Resize image to fit canvas while maintaining aspect ratio
        height, width = image.shape[:2]
        canvas_width = 800
        canvas_height = 600
        
        # Calculate scaling factor
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
        
        # Update canvas
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width//2, new_height//2, image=photo, anchor='center')
        self.canvas.image = photo  # Keep a reference!

if __name__ == "__main__":
    root = tk.Tk()
    app = FODDetector(root)
    root.mainloop()