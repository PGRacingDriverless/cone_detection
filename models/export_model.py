from ultralytics import YOLO

# Load the pretrained model
model = YOLO("cones_v2.pt")

# Export the model to ONNX format
model.export(format="onnx", opset=16, imgsz=640, simplify=True, dynamic=False)
