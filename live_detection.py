
import time
import threading

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
from picamera2 import Picamera2

# COCO classes relevant to the V2X project
CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

CONFIDENCE = 0.40
MODEL_SIZE = 640

# Load YOLOv8n ONNX model
session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# Configure Raspberry Pi Camera Module 3
camera = Picamera2()
config = camera.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(config)

# Web server for viewing the camera over SSH
app = Flask(__name__)
latest_jpeg = None
frame_lock = threading.Lock()


@app.route("/")
def home():
    return """
    <html>
    <head><title>BlindSpot V2X</title></head>
    <body style="background:#111;color:white;text-align:center;
                 font-family:Arial">
        <h1>BlindSpot V2X - Live Detection</h1>
        <img src="/video" style="width:90%;max-width:900px">
        <p>Raspberry Pi 4 | YOLOv8n | Camera Module 3</p>
    </body>
    </html>
    """


@app.route("/video")
def video():
    def generate():
        while True:
            with frame_lock:
                jpeg = latest_jpeg

            if jpeg is None:
                time.sleep(0.1)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_detection():
    global latest_jpeg

    camera.start()
    print("Camera and YOLO running. Press Ctrl+C to stop.")

    try:
        while True:
            frame = camera.capture_array()
            height, width = frame.shape[:2]

            # Letterbox image to 640x640
            scale = min(
                MODEL_SIZE / width,
                MODEL_SIZE / height
            )

            new_w = round(width * scale)
            new_h = round(height * scale)

            resized = cv2.resize(frame, (new_w, new_h))

            image = np.full(
                (MODEL_SIZE, MODEL_SIZE, 3),
                114,
                dtype=np.uint8
            )

            pad_x = (MODEL_SIZE - new_w) // 2
            pad_y = (MODEL_SIZE - new_h) // 2

            image[
                pad_y:pad_y + new_h,
                pad_x:pad_x + new_w
            ] = resized

            # Convert BGR image to RGB for YOLO
            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            tensor = rgb.transpose(2, 0, 1)
            tensor = np.ascontiguousarray(
                tensor, dtype=np.float32
            )[None] / 255.0

            start = time.perf_counter()

            output = session.run(
                None,
                {input_name: tensor}
            )[0]

            elapsed = time.perf_counter() - start

            # YOLOv8 output: boxes + class scores
            predictions = output[0].T

            boxes = []
            scores = []
            class_ids = []

            for prediction in predictions:
                class_id = int(
                    np.argmax(prediction[4:])
                )

                confidence = float(
                    prediction[4 + class_id]
                )

                if class_id not in CLASSES:
                    continue

                if confidence < CONFIDENCE:
                    continue

                cx, cy, w, h = prediction[:4]

                x = int(
                    (cx - w / 2 - pad_x) / scale
                )
                y = int(
                    (cy - h / 2 - pad_y) / scale
                )

                box_w = int(w / scale)
                box_h = int(h / scale)

                boxes.append([x, y, box_w, box_h])
                scores.append(confidence)
                class_ids.append(class_id)

            # Non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                CONFIDENCE,
                0.45
            )

            detections = []

            for i in np.array(
                indices, dtype=int
            ).flatten():

                name = CLASSES[class_ids[i]]
                confidence = scores[i]

                detections.append(
                    f"{name} ({confidence:.0%})"
                )

                x, y, w, h = boxes[i]

                # Keep bounding boxes inside image
                x1 = max(0, min(width - 1, x))
                y1 = max(0, min(height - 1, y))
                x2 = max(0, min(width - 1, x + w))
                y2 = max(0, min(height - 1, y + h))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{name} {confidence:.0%}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            fps = 1 / elapsed if elapsed > 0 else 0

            # Display inference performance
            cv2.putText(
                frame,
                f"Inference: {elapsed * 1000:.0f} ms"
                f" | {fps:.2f} FPS",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            print(
                f"Detection: {', '.join(detections) or 'None'} "
                f"| Inference: {elapsed * 1000:.0f} ms "
                f"| FPS: {fps:.2f}",
                flush=True
            )

            # Encode annotated frame for browser
            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if success:
                with frame_lock:
                    latest_jpeg = encoded.tobytes()

    finally:
        camera.stop()
        camera.close()


if __name__ == "__main__":
    worker = threading.Thread(
        target=run_detection,
        daemon=True
    )
    worker.start()

    app.run(
        host="127.0.0.1",
        port=5000,
        threaded=True,
        use_reloader=False
    )
