import cv2
import numpy as np
import time

print("BlindSpot V2X - Detection System")
print("OpenCV version:", cv2.__version__)

def detect_hazard(distance):
    if distance < 30:
        return "DANGER - STOP"
    elif distance < 60:
        return "WARNING - SLOW DOWN"
    else:
        return "SAFE"

distances = [100, 75, 50, 25, 10]

for distance in distances:
    status = detect_hazard(distance)
    print(f"Distance: {distance} cm | {status}")
    time.sleep(1)
