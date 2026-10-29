import cv2
import torch
import time
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights
)

# Load the pretrained neural network
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# COCO object names
categories = weights.meta["categories"]

# Open laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Cannot open webcam")
    raise SystemExit

print("SSD MobileNet running. Press Q to quit.")

inference_times = []
frame_count = 0

while True:
    success, frame = camera.read()

    if not success:
        break

    # Convert webcam image to a PyTorch tensor
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    # Run neural network and measure inference time
    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model([image])[0]

    inference_ms = (time.perf_counter() - start) * 1000
    inference_times.append(inference_ms)
    frame_count += 1

    # Extract detected objects
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    for box, label, score in zip(boxes, labels, scores):
        confidence = score.item()

        if confidence < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        name = categories[label.item()]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display inference time and estimated FPS
    fps = 1000 / inference_ms

    cv2.putText(
        frame,
        f"Inference: {inference_ms:.1f} ms | FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("SSD MobileNet - BlindSpot V2X", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

# Print benchmark summary
if inference_times:
    average_ms = sum(inference_times) / len(inference_times)

    print("\n--- SSD MobileNet Benchmark ---")
    print(f"Frames processed: {frame_count}")
    print(f"Average inference: {average_ms:.1f} ms")
    print(f"Estimated inference FPS: {1000 / average_ms:.2f}")