
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Dataset locations
DATASET = Path.home() / "Downloads" / "COCO"
IMAGES = DATASET / "val2017"
ANNOTATIONS = DATASET / "annotations" / "instances_val2017.json"

# Start with 100 images to keep the test manageable
NUM_IMAGES = 100
DEVICE = "cpu"

torch.set_num_threads(4)

coco = COCO(str(ANNOTATIONS))
image_ids = sorted(coco.getImgIds())[:NUM_IMAGES]

# Map the 80 COCO model classes to COCO category IDs
category_ids = sorted(coco.getCatIds())
assert len(category_ids) == 80

models = {
    "SSD MobileNet": ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval().to(DEVICE),

    "YOLOv8 Medium": YOLO("yolov8m.pt"),

    "Faster R-CNN": fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).eval().to(DEVICE),
}

# Torchvision uses COCO labels with gaps in their numbering
torchvision_categories = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

def add_detection(results, image_id, category_id, box, score):
    x1, y1, x2, y2 = [float(v) for v in box]
    results.append({
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "score": float(score),
    })

for name, model in models.items():
    print(f"\n--- Testing {name} ---")

    detections = []
    inference_times = []

    # Warm up the model so initial setup does not affect timing
    first_image = coco.loadImgs(image_ids[0])[0]
    first_path = IMAGES / first_image["file_name"]
    warmup = Image.open(first_path).convert("RGB")

    with torch.inference_mode():
        if name == "YOLOv8 Medium":
            model.predict(np.array(warmup), device="cpu", verbose=False)
        else:
            model([F.to_tensor(warmup).to(DEVICE)])

    for index, image_id in enumerate(image_ids, 1):
        image_info = coco.loadImgs(image_id)[0]
        image = Image.open(IMAGES / image_info["file_name"]).convert("RGB")

        start = time.perf_counter()

        with torch.inference_mode():
            if name == "YOLOv8 Medium":
                prediction = model.predict(
                    np.array(image),
                    device="cpu",
                    verbose=False,
                    conf=0.001,
                    iou=0.7,
                    imgsz=640,
                    max_det=300,
                )[0]
            else:
                prediction = model([F.to_tensor(image).to(DEVICE)])[0]

        elapsed = (time.perf_counter() - start) * 1000
        inference_times.append(elapsed)

        if name == "YOLOv8 Medium":
            for box in prediction.boxes:
                cls = int(box.cls.item())
                add_detection(
                    detections, image_id, category_ids[cls],
                    box.xyxy[0].tolist(), box.conf.item()
                )
        else:
            for box, label, score in zip(
                prediction["boxes"],
                prediction["labels"],
                prediction["scores"],
            ):
                label = int(label.item())
                if label not in torchvision_categories:
                    continue
                add_detection(
                    detections, image_id, label,
                    box.tolist(), score.item()
                )

        if index % 10 == 0:
            print(f"Processed {index}/{NUM_IMAGES} images")

    average_ms = sum(inference_times) / len(inference_times)
    estimated_fps = 1000 / average_ms

    print(f"Average processing time: {average_ms:.1f} ms")
    print(f"Estimated processing FPS: {estimated_fps:.2f}")

    if detections:
        results_file = Path("benchmarks") / (
            name.lower().replace(" ", "_").replace("-", "_")
            + "_detections.json"
        )
        results_file.parent.mkdir(exist_ok=True)
        results_file.write_text(json.dumps(detections))

        coco_results = coco.loadRes(detections)
        evaluator = COCOeval(coco, coco_results, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        print(f"COCO mAP (0.50:0.95): {evaluator.stats[0]:.4f}")

print("\nBenchmark finished!")
