import cv2
import argparse
import os.path

try:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input video")
    args = vars(ap.parse_args())
    path_to_file = args["image"]

    if not os.path.exists(path_to_file):
        raise FileNotFoundError("File not found")

    cap = cv2.VideoCapture(path_to_file)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    out = cv2.VideoWriter(r'output.mp4', fourcc, cap_fps, (cap_width, cap_height), 0)

    number_frame = 0
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if ret:
            out_frame = cv2.resize(frame, (cap_width, cap_height))
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)

            number_frame += 1
            text = str(number_frame)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            cv2.putText(out_frame, text, (0, text_size[1]), font, font_scale, (0, 0, 255), thickness)

            out.write(out_frame)
        else:
            break

    cap.release()
    out.release()
    print("A video file named output.mp4 has been created")
except Exception as e:
    print(e)
