# Importing all the required libraries
import cv2 as cv
import numpy as np
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


proto = os.path.join(BASE_DIR, "..", "models", "deploy.prototxt")
model = os.path.join(BASE_DIR, "..", "models", "mobilenet_iter_73000.caffemodel")

if not os.path.exists(proto) or not os.path.exists(model):
    print("Model files are missing")
    sys.exit(1)

# Using MobileNet SSD as a pretrained model
net = cv.dnn.readNetFromCaffe(proto, model)

# Normalising the input image
shape = (300, 300)
mean = (127.5, 127.5, 127.5)
scale = 0.007843

# Class list
classNames = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
    4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

# Detect yellow colour in ROI
def yellow(roi):
    if roi is None or roi.size == 0:
        return False
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    low = np.array([20, 100, 100])
    high = np.array([35, 255, 255])
    mask = cv.inRange(hsv, low, high)
    yp = cv.countNonZero(mask)
    total = roi.shape[0] * roi.shape[1]
    return (yp / total) * 100 > 18  # Threshold percentage


# Detect yellow car using MobileNet SSD
def yellow_car(imgPath):
    img = cv.imread(imgPath)
    if img is None:
        print("No such image found")
        return

    h, w = img.shape[:2]

    blob = cv.dnn.blobFromImage(
        img,
        scalefactor=scale,
        size=shape,
        mean=mean,
        swapRB=True
    )

    net.setInput(blob)
    out = net.forward()

    yellow_found = False

    for i in range(out.shape[2]):
        conf = out[0, 0, i, 2]
        if conf < 0.5:
            continue

        cid = int(out[0, 0, i, 1])
        if classNames.get(cid, "") != "car":
            continue

        # Box coordinates
        x1, y1, x2, y2 = out[0, 0, i, 3:7]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        # Crop ROI
        roi = img[y1:y2, x1:x2]

        # Check yellow colour
        if yellow(roi):
            yellow_found = True
            color = (0, 255, 255)  # Yellow bounding box
            label = "YELLOW CAR"
            print("Yellow Car Detected!")
        else:
            color = (0, 255, 0)
            label = "Car"

        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv.putText(img, label, (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Resize and show output
    tmp = cv.resize(img, (0, 0), fx=0.7, fy=0.7)
    cv.imshow("Yellow car detection", tmp)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if not yellow_found:
        print("No yellow car found")


#image path
img_path = os.path.join(BASE_DIR, "..", "images", "yellow-beetle.jpg")

yellow_car(img_path)
