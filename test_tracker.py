import cv2
import dlib
from time import time


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global box

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if paused:
        if event == cv2.EVENT_LBUTTONDOWN:
            box[0], box[1] = x, y

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            box[2], box[3] = x, y

        # # draw a rectangle around the region of interest
            cv2.rectangle(img, (box[0], box[1]), (x, y), (0, 255, 0), 2)
            cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

cap = cv2.VideoCapture('videos/input.mp4')
_, img = cap.read()
paused = False
tracking = False
tracker = dlib.correlation_tracker()
box = [None, None, None, None]

while True:
    if not paused:
        _, img = cap.read()
        H, W = img.shape[:2]
        img = cv2.resize(img,(W//5,H//5))
        H, W = H//5, W//5

        # start = time()
        if tracking:
            tracker.update(img)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow("image", img)    
        # print(time()-start)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("r"):
        paused = False
    elif key == ord("p"):
        paused = True
        tracking = False
    elif key == ord("t"):
        paused = False
        tracking = True
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        tracker.start_track(img, rect)
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()