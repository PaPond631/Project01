import cv2
import random
import time
def playVideo():
    cap = cv2.VideoCapture("GR10.mp4")
    width = int(cap.get(3))
    height = int(cap.get(4))

    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Player", width, height)

    while(cap.isOpened()):
        success, frame = cap.read()
        if success:
            cv2.imshow('Video Player', frame)
            quitButton = cv2.waitKey(25) & 0xFF == ord('q')
            closeButton = cv2.getWindowProperty('Video Player', cv2.WND_PROP_VISIBLE) < 1
            if quitButton or closeButton:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


while True:
    # พี่ลองสร้างเื่อนไขให้มันเช็คถ้าเลขเป็น 1 ให้แสดงวิดีโอ ของปอนก็คือเงื่อนไขในภาพมีคนใส่แว่นนะ
    n = random.randint(1, 10)
    print("N = ",n)
    if n == 1:
        playVideo()
    time.sleep(1)