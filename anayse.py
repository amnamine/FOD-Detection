from ultralytics import YOLO

# Load the model using YOLO
model = YOLO('fod.pt')

# Print number of classes
print(f"Number of classes in FOD dataset: {len(model.names)}")
