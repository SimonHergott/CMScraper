from ultralytics import YOLO

DATA_YAML = "/home/simon/CMScraper/training/merged_dataset/data.yaml"
MODEL = "yolo11n.pt"
EPOCHS = 100
IMGSZ = 1280
BATCH = 2
DEVICE = "cuda"
PROJECT = "/home/simon/CMScraper/runs"
NAME = "cms_detector"

model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT,
    name=NAME,
    amp=False,
)
