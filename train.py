from ultralytics import YOLO

# Load model YOLOv8 pretrained
model = YOLO("yolov8n.pt")  # kamu juga bisa pakai yolov8s.pt, dll

# Latih model
model.train(data=r"D:\REPO PROJECT\utspcd\dataset\data.yaml", epochs=50, imgsz=640, batch=16)

