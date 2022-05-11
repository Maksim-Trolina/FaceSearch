import cv2
import numpy as np
import argparse
import os.path


def main():
    try:
        path_to_file = get_path_to_file()

        if not os.path.exists(path_to_file):
            raise FileNotFoundError("File not found")

        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(path_to_file)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        color = np.random.randint(0, 255, (100, 3))

        mask = None
        p0 = None
        old_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            crop_frame, found = get_crop_frame(frame, faces)
            if not found:
                continue
            if mask is None:
                mask, p0, old_gray = init_start_frame(crop_frame)
                continue

            frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

            if not frame_equal(old_gray, frame_gray):
                old_gray = frame_gray
                continue
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                crop_frame = cv2.circle(crop_frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv2.add(crop_frame, mask)
            display_frame(img)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cap.release()
        cv2.destroyAllWindows()
        print("A video file named output.mp4 has been created")

    except Exception as e:
        print(e)


def get_crop_frame(frame, faces):
    results = faces.detectMultiScale(frame, scaleFactor=1.9, minNeighbors=5)
    if len(results) == 0:
        return frame, False
    x, y, w, h = results[0]
    crop_frame = frame[y:y + h, x:x + w]
    return crop_frame, True


def get_path_to_file():
    path_to_file = "videoplayback.mp4"
    return path_to_file


def frame_equal(frame1, frame2):
    return frame1.shape[0] == frame2.shape[0] and frame1.shape[1] == frame2.shape[1]


def display_frame(frame):
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    return k


def init_start_frame(frame):
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
    mask = np.zeros_like(frame)
    return p0, mask, gray_frame


main()
