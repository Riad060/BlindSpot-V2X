\# BlindSpot V2X — Neural Network Benchmarks



\## Objective

Evaluate different object detection neural networks for a cooperative V2X hazard detection system, comparing processing speed and suitability for deployment on a Raspberry Pi 4.



\## Test 1: YOLOv8 Medium

\- Platform: Windows laptop, CPU

\- Input: Laptop webcam

\- Reported inference time: 460.4 ms

\- Estimated inference FPS: 2.17

\- Status: Successfully detected people and surrounding objects



\## Test 2: SSDlite320 MobileNet V3

\- Platform: Windows laptop, CPU

\- Input: Laptop webcam

\- Frames processed: 1,462

\- Average inference time: 46.8 ms

\- Estimated inference FPS: 21.39

\- Status: Successfully ran object detection



\## Limitations

These are preliminary tests using different measurement methods. A controlled benchmark using identical images, input settings and measurement procedures is required before drawing performance conclusions.



\## Next Steps

1\. Test a larger object detection neural network.

2\. Compare all three architectures on a common dataset.

3\. Evaluate suitable models on Raspberry Pi 4.

4\. Record results and discuss the speed–accuracy trade-off.

