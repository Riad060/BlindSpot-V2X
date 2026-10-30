
import cv2
import torch
import time
import numpy as np

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

# Load pretrained Faster R-CNN model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
model.to("cpu")

# COCO object labels
labels = weights.meta["categories"]
preprocess = weights.transforms()

# Open laptop webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

inference_times = []
frame_count = 0

print("Faster R-CNN running. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert camera frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = preprocess(
        __import__("PIL").Image.fromarray(rgb)
    )

    # Measure neural network inference time
    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model([image])[0]

    elapsed = time.perf_counter() - start
    inference_times.append(elapsed)
    frame_count += 1

    # Draw detected objects
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    classes = prediction["labels"].cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        label = labels[int(cls)]

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Faster R-CNN - BlindSpot V2X", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Print benchmark results
if inference_times:
    avg_ms = np.mean(inference_times) * 1000
    fps = 1000 / avg_ms

    print("\n--- Faster R-CNN Benchmark ---")
    print(f"Frames processed: {frame_count}")
    print(f"Average inference: {avg_ms:.1f} ms")
    print(f"Estimated inference FPS: {fps:.2f}")
