# Importing all the required libraries
import cv2 as cv
import numpy as np
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Using MobileNet SSD as a pretrained model
proto = os.path.join(BASE_DIR, "..", "models", "deploy.prototxt")
model = os.path.join(BASE_DIR, "..", "models", "mobilenet_iter_73000.caffemodel")

if not os.path.exists(proto) or not os.path.exists(model):
    print("Model files are missing")
    sys.exit(1)

net = cv.dnn.readNetFromCaffe(proto, model)

# Normalising the input image
shape = (300, 300)
mean = (127.5, 127.5, 127.5)
scale = 0.007843

# Objects which can be identified using MobileNet SSD
classNames = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
    4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

# Creating Trackers 
trackerList = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
trackerType = 'CSRT'   


def createtracker(tname):
    """Create a tracker based on tracker name"""
    tname = tname.upper()
    if tname == 'BOOSTING':
        return cv.legacy.TrackerBoosting_create()
    elif tname == 'MIL':
        return cv.TrackerMIL_create()
    elif tname == 'KCF':
        return cv.legacy.TrackerKCF_create()
    elif tname == 'TLD':
        return cv.legacy.TrackerTLD_create()
    elif tname == 'MEDIANFLOW':
        return cv.legacy.TrackerMedianFlow_create()
    elif tname == 'MOSSE':
        return cv.legacy.TrackerMOSSE_create()
    elif tname == 'CSRT':
        return cv.TrackerCSRT_create()
    else:
        print("No Such Tracker name:", tname)
        print("Tracker Available", trackerList)
        return None


# Intersection over union function
def IOU(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return inter / float(area1 + area2 - inter)


# Function for detecting yellow color
def yellow(roi):
    if roi is None or roi.size == 0:
        return False

    roi_blur = cv.GaussianBlur(roi, (5, 5), 0)
    hsv = cv.cvtColor(roi_blur, cv.COLOR_BGR2HSV)

    low = np.array([15, 80, 80], dtype=np.uint8)
    high = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv.inRange(hsv, low, high)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

    yp = cv.countNonZero(mask)
    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return False

    per = (yp / total) * 100.0
    return per > 18.0     


# Function to detect and track yellow cars in video
def yellow_car(videoPath):
    cap = cv.VideoCapture(videoPath)
    if not cap.isOpened():
        print("No such video path available")
        return

    yellowCars = []
    oldCars = []
    nextId = 0
    frameNo = 0

    orb = cv.ORB_create(nfeatures=300)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    slam_orb = cv.ORB_create(nfeatures=800)
    prev_gray = None
    slam_R = np.eye(3, dtype=np.float64)
    slam_t = np.zeros((3, 1), dtype=np.float64)

    detectGap = 7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameNo += 1
        h, w = frame.shape[:2]

        gray_full = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if prev_gray is not None:
            kp1, des1 = slam_orb.detectAndCompute(prev_gray, None)
            kp2, des2 = slam_orb.detectAndCompute(gray_full, None)

            if des1 is not None and des2 is not None and len(kp1) > 8 and len(kp2) > 8:
                slam_bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                matches = slam_bf.match(des1, des2)

                if len(matches) >= 8:
                    matches = sorted(matches, key=lambda m: m.distance)
                    good_matches = matches[:200]

                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    F, mask_F = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.0, 0.99)
                    if F is not None and F.shape == (3, 3) and mask_F is not None:
                        inliers = mask_F.ravel() == 1
                        pts1_in = pts1[inliers]
                        pts2_in = pts2[inliers]

                        if pts1_in.shape[0] >= 8:
                            f = 0.9 * max(w, h)
                            K = np.array([[f, 0, w / 2.0],
                                          [0, f, h / 2.0],
                                          [0, 0, 1]], dtype=np.float64)

                            E = K.T @ F @ K

                            try:
                                _, R, t, _ = cv.recoverPose(E, pts1_in, pts2_in, K)
                                slam_t = slam_t + slam_R @ t
                                slam_R = R @ slam_R

                                text = f"SLAM pose x:{slam_t[0,0]:.2f} z:{slam_t[2,0]:.2f}"
                                cv.putText(frame, text, (10, h - 20),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            except cv.error:
                                pass

        prev_gray = gray_full.copy()

        # Update tracker
        updatedCars = []
        for item in yellowCars:
            tr = item["tracker"]
            if tr is None:
                continue

            ok, box = tr.update(frame)
            item["age"] += 1

            if ok:
                x, y, ww, hh = [int(z) for z in box]

                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                ww = max(1, min(ww, w - x))
                hh = max(1, min(hh, h - y))

                item["bbox"] = (x, y, ww, hh)
                item["miss"] = 0

                cv.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 255), 2)
                label = "Yellow Car {}".format(item["id"])
                cv.putText(frame, label, (x, y - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                item["miss"] += 1

                if item["miss"] > 10:
                    if item.get("desc") is not None:
                        oldCars.append({
                            "id": item["id"],
                            "desc": item["desc"]
                        })
                    continue

            updatedCars.append(item)

        yellowCars = updatedCars

        runDetect = (frameNo % detectGap == 1) or (len(yellowCars) == 0)

        if runDetect:
            blob = cv.dnn.blobFromImage(frame,
                                        scalefactor=scale,
                                        size=shape,
                                        mean=mean,
                                        swapRB=True)
            net.setInput(blob)
            out = net.forward()

            for i in range(out.shape[2]):
                conf = out[0, 0, i, 2]
                if conf < 0.5:
                    continue

                cid = int(out[0, 0, i, 1])
                if classNames.get(cid, "") != "car":
                    continue

                x1, y1, x2, y2 = out[0, 0, i, 3:7]
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                ww = x2 - x1
                hh = y2 - y1

                if ww < 25 or hh < 25:
                    continue

                boxDet = [x1, y1, x2, y2]
                roi = frame[y1:y2, x1:x2]

                if not yellow(roi):
                    continue

                dup = False
                for item in yellowCars:
                    bx, by, bw, bh = item["bbox"]
                    tb = [bx, by, bx + bw, by + bh]

                    iou = IOU(boxDet, tb)

                    cx1 = (x1 + x2) // 2
                    cy1 = (y1 + y2) // 2
                    cx2 = bx + bw // 2
                    cy2 = by + bh // 2
                    dist = np.hypot(cx1 - cx2, cy1 - cy2)

                    size1 = ww * hh
                    size2 = bw * bh
                    size_ratio = min(size1, size2) / max(size1, size2)

                    if iou > 0.40 or dist < 60 or size_ratio > 0.70:
                        dup = True
                        break

                if dup:
                    continue

                gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                kp, des_new = orb.detectAndCompute(gray_roi, None)

                newCar = True
                if des_new is not None and len(oldCars) > 0:
                    for prev in oldCars:
                        prev_des = prev["desc"]
                        if prev_des is None:
                            continue

                        matches = bf.match(des_new, prev_des)
                        if len(matches) == 0:
                            continue

                        distList = [m.distance for m in matches]
                        avgDist = sum(distList) / len(distList)
                        good = [m for m in matches if m.distance < 50]

                        if avgDist < 60 and len(good) >= 10:
                            newCar = False

                            tr_old = createtracker(trackerType)
                            if tr_old is not None:
                                tr_old.init(frame, (x1, y1, ww, hh))
                                yellowCars.append({
                                    "id": prev["id"],
                                    "tracker": tr_old,
                                    "bbox": (x1, y1, ww, hh),
                                    "age": 0,
                                    "miss": 0,
                                    "desc": prev_des
                                })
                            break

                if not newCar:
                    continue

                print("Yellow Car!")

                tr = createtracker(trackerType)
                if tr is None:
                    continue

                tr.init(frame, (x1, y1, ww, hh))

                yellowCars.append({
                    "id": nextId,
                    "tracker": tr,
                    "bbox": (x1, y1, ww, hh),
                    "age": 0,
                    "miss": 0,
                    "desc": des_new
                })

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv.putText(frame, "New Yellow Car !{}".format(nextId),
                           (x1, y1 - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                nextId += 1

        # Display frame
        tmp = cv.resize(frame, (0, 0), fx=0.55, fy=0.55)
        cv.imshow("Yellow Car Tracking + SLAM Front-End", tmp)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



video_path = os.path.join(
    BASE_DIR,
    "..",
    "Videos",
    "Yellow cars descend on Cotswold village in support of local [LPxc8KYO3Jk].mp4"
)

yellow_car(video_path)
