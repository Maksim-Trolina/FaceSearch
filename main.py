import mediapipe as mp
import cv2


def get_faces(mp_face_detector, frame):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mp_face_detector.process(frameRGB)
    return faces

def draw_face_rectangles(faces, frame):
    image_height, image_width, _ = frame.shape
    for face_no, face in enumerate(faces.detections):
        face_bbox = face.location_data.relative_bounding_box
        x = int(face_bbox.xmin * image_width)
        y = int(face_bbox.ymin * image_height)
        height = int(face_bbox.height * image_height)
        width = int(face_bbox.width * image_width)
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                  color=(0, 0, 255), thickness=3)

def main():
    cap = cv2.VideoCapture(0)
    mp_face_detection = mp.solutions.face_detection
    mp_face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
    while True:
        success, frame = cap.read()
        if not success:
            continue
        faces = get_faces(mp_face_detector, frame)
        if faces.detections:
            draw_face_rectangles(faces, frame)
        cv2.imshow('Face detection', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

main()