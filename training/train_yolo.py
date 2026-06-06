from ultralytics import YOLO

DATA_YAML = "CMScraper/training/merged_dataset/data.yaml"
MODEL = "yolo11l.pt"
EPOCHS = 100
IMGSZ = 1280
BATCH = 16
DEVICE = "cuda"
PROJECT = "CMScraper/runs"
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
