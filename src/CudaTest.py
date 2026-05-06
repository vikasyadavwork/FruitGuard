from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolo11n.pt")

results = model(
    source=0,       # webcam
    device=0,       # GPU 0
    show=True,
    verbose=False
)