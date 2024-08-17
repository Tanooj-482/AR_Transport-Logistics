import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import requests
from PIL import Image, ImageTk

# Initialize Mediapipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to overlay an accessory (e.g., clothing) on the image
def overlay_image(background, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]
    alpha_overlay = overlay[:, :, 3] / 255.0  # Get alpha channel for transparency
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] + 
                                       alpha_background * background[y:y+h, x:x+w, c])

    return background

# GUI Class for Product Selection
class ProductSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AR Try-On Product Selector")
        self.geometry("400x200")
        
        self.label = tk.Label(self, text="Select a product to try on:")
        self.label.pack(pady=20)

        self.product_button = tk.Button(self, text="Choose Product", command=self.select_product)
        self.product_button.pack(pady=20)

        self.selected_product = None

    def select_product(self):
        # Open file dialog to select a product image
        file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if file_path:
            self.selected_product = file_path
            self.label.config(text=f"Selected: {file_path.split('/')[-1]}")

# Function to simulate loading a product catalog (can be connected to an API)
def load_product_catalog():
    # Simulated catalog for demonstration purposes
    catalog = [
        {"name": "Hat", "image": "hat.png"},
        {"name": "Sunglasses", "image": "sunglasses.png"}
        # Add more products here
    ]
    return catalog

# Initialize GUI and Product Catalog
product_selector = ProductSelector()
product_catalog = load_product_catalog()

# Run the GUI in a separate thread or the main thread
product_selector.mainloop()

# Check if a product has been selected
if product_selector.selected_product:
    product_image = cv2.imread(product_selector.selected_product, cv2.IMREAD_UNCHANGED)
else:
    print("No product selected. Exiting...")
    exit()

# Open webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect pose landmarks
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        # Get key points (e.g., nose for placing the hat)
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_coords = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
        
        # Overlay the selected product (e.g., clothing) on the body
        frame = overlay_image(frame, product_image, nose_coords)
    
    # Show the frame
    cv2.imshow("AR Try-On", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
