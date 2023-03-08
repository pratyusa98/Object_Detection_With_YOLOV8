from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = YOLO('../Yolo-Weights/best.pt')

results = model("Images/apple.jpg", show=True)

cv2.waitKey(0)

# For Webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)


