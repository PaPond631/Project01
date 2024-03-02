from ultralytics import YOLO
import cv2
import math
import numpy as np
from ffpyplayer.player import MediaPlayer

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO("../YOLO-Weights/detect.pt")
classNames = ['backpack', 'glasses','iphone','no_glasses']


def getVideoSource(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def running_videos(sourcePath):

    camera = getVideoSource(sourcePath, 720, 480)
    player = MediaPlayer(sourcePath)

    while True:
        ret, frame = camera.read()
        audio_frame, val = player.get_frame()

        if (ret == 0):
            print("End of video")
            break

        frame = cv2.resize(frame, (720, 480))
        cv2.imshow('Camera', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyWindow('Camera')


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    glasses_detected = False
    iphone_detected = False
    backpack_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'

            if class_name == 'glasses' and conf >= 0.7:
                glasses_detected = True

            if class_name == 'iphone' and conf >= 0.7:
                iphone_detected = True
            
            if class_name == 'backpack' and conf >=0.7:
                backpack_detected = True

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [
                          255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1-2), 0, 1,
                        [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    out.write(img)
    cv2.imshow("Image", img)

    # Break the loop if 'q' key is pressed or when glasses are detected
    if glasses_detected:
        running_videos("C:\YOLOv8\Running_YOLOv8_Webcam\Ophtus.mp4")
        cv2.destroyWindow("Image")
    if iphone_detected:
        running_videos("C:\YOLOv8\Videos\iphone.mp4")
        cv2.destroyWindow("Image")
    if backpack_detected:
        running_videos("C:\YOLOv8\Videos\ibackpack.mp4")
        cv2.destroyWindow("Image")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
