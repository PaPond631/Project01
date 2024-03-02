from ultralytics import YOLO
import cv2
import math
import numpy as np
from ffpyplayer.player import MediaPlayer
import time
from datetime import timedelta
import datetime
import pytz

glass_advertise_numb = 0
phone_advertise_numb = 0
backpack_advertise_numb = 0

cap = cv2.VideoCapture(0)

total_Glass = 0
total_Back = 0
total_iphone = 0

total_Glass_interest = 0
total_Back_interest = 0
total_iphone_interest = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

count_view_glasses_Opthus = 0
count_view_glasses_TopCHA = 0
count_view_iphone = 0
count_view_zflip = 0
count_view_backpack = 0
count_view_Handback = 0

current_video = None
current_time = datetime.datetime.utcnow()

thai_time = current_time.astimezone(datetime.timezone(datetime.timedelta(hours=14)))

thai_date = thai_time.strftime("%d-%m-%Y")
time_thai = thai_time.strftime("%H:%M")

timezone_th = pytz.timezone('Asia/Bangkok')


out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 5, (frame_width, frame_height))

model = YOLO("../YOLO-Weights/detect.pt")
classNames = ['backpack', 'glasses','iphone','no_glasses']

interest_counts = {
    "C:\YOLOv8\Running_YOLOv8_Webcam\Ophtus.mp4": 0,
    "C:\YOLOv8\Videos\Top_Charoen.mp4": 0,
    "C:\YOLOv8\Videos\iphone.mp4": 0,
    "C:\YOLOv8\Videos\Galaxy Z Flip4.mp4": 0,
    "C:\YOLOv8\Videos\ibackpack.mp4": 0,
    "C:\YOLOv8\Videos\Handbag.mp4": 0
}



def getVideoSource(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def running_videos(sourcePath):

    global current_video
    current_video = sourcePath

    camera = getVideoSource(sourcePath, 720, 480)
    player = MediaPlayer(sourcePath)
    start_time_interest = None
    interest_recorded = False

    while True:
        ret, frame = camera.read()
        success, img = cap.read()
        audio_frame, val = player.get_frame()

        results = model(img, stream=True)
        glasses_detected = False
        iphone_detected = False
        backpack_detected = False
        no_glasses_detected = False

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

                if class_name == 'no_glasses' and conf >=0.7:
                    no_glasses_detected = True

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [
                              255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1-2), 0, 1,
                            [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                
        if glasses_detected or iphone_detected or backpack_detected or no_glasses_detected:
            if start_time_interest is None:
                start_time_interest = time.time()
        else:
            start_time_interest = None

        if not interest_recorded and start_time_interest is not None:
            if time.time() - start_time_interest >=3:
                with open("test01.txt", "a") as f:
                    f.write(f"1\n")
                    
            
                interest_counts[current_video] += 1            
                start_time_interest = None
                interest_recorded = True
        

        if (ret == 0):
            with open("test01.txt", "a") as f:
                f.write(f"0\n")
            print("End of video")
            break

        frame = cv2.resize(frame, (620, 380))
        cv2.imshow('Camera', frame)
        cv2.imshow('Image', img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            with open("test01.txt", "a") as f:
                f.write(f"\n")
            break


    camera.release()
    cv2.destroyWindow('Camera')


#START
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

            elif class_name == 'iphone' and conf >= 0.7:
                iphone_detected = True
            
            elif class_name == 'backpack' and conf >=0.7:
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

    # NAME, WATCH, TIME, DATE, INTEREST
    if glasses_detected:
        total_Glass += 1      
        current_time_th = datetime.datetime.now(timezone_th)
        formatted_time = current_time_th.strftime("%H:%M") 
        
        if glass_advertise_numb == 0:
            count_view_glasses_Opthus += 1
            glass_advertise_numb += 1

            with open("test01.txt", "a") as f:
                f.write(f"Ophtus,1,{formatted_time},{thai_date},")
            running_videos("C:\YOLOv8\Running_YOLOv8_Webcam\Ophtus.mp4")
            cv2.destroyWindow("Image")
        else: 
            count_view_glasses_TopCHA += 1
            glass_advertise_numb -= 1
            with open("test01.txt", "a") as f:
                f.write(f"TopCHA,1,{formatted_time},{thai_date},")
            running_videos("C:\YOLOv8\Videos\Top_Charoen.mp4")
            cv2.destroyWindow("Image")

    if iphone_detected:
        total_iphone += 1
        current_time_th = datetime.datetime.now(timezone_th)
        formatted_time = current_time_th.strftime("%H:%M")

        if phone_advertise_numb == 0:
            count_view_iphone += 1
            phone_advertise_numb += 1 

            with open("test01.txt", "a") as f:
                f.write(f"Apple,1,{formatted_time},{thai_date},")
            running_videos("C:\YOLOv8\Videos\iphone.mp4")
            cv2.destroyWindow("Image")
        else:
            count_view_zflip += 1
            phone_advertise_numb -= 1
            with open("test01.txt", "a") as f:
                f.write(f"Z_Flip,1,{formatted_time},{thai_date},")
            running_videos("C:\YOLOv8\Videos\Galaxy Z Flip4.mp4")
            cv2.destroyWindow("Image")

    if backpack_detected:
        total_Back += 1
        current_time_th = datetime.datetime.now(timezone_th)
        formatted_time = current_time_th.strftime("%H:%M")

        if backpack_advertise_numb == 0:
            count_view_backpack += 1
            backpack_advertise_numb += 1

            with open("test01.txt", "a") as f:
                f.write(f"BackPack,1,{formatted_time},{thai_date},")
            running_videos("C:\YOLOv8\Videos\ibackpack.mp4")
            cv2.destroyWindow("Image")
        else:
            count_view_Handback += 1
            backpack_advertise_numb -= 1
            with open("test01.txt", "a") as f:
                f.write(f"HandBack,1,{formatted_time},{thai_date},")
            running_videos("C:\YOLOv8\Videos\Handbag.mp4")
            cv2.destroyWindow("Image")

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break


with open("test01.txt", "a") as f:
    f.write("\n")

out.release()
cap.release()
cv2.destroyAllWindows()
