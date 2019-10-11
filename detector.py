'''
python3 detector.py -p my_ssd/MobileNetSSD_deploy.prototxt -m my_ssd/mobilenet_iter_1800.caffemodel -i videos/input.mp4 -o result.avi
'''

import cv2
import imutils
import time
import glob
import numpy as np
import dlib
import argparse
import math as m

from my_tracker.tracker import Tracker, Counter

# door position, [left, right, top, bottom]
door = []
# flag for initial door
init = 0


def detect(net, img, min_confidence, W, H):
    # list of tracking objects detected by detector
    ls_trackers = []

    # put image to net and run dectector
    blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over all detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            # class_id = int(detections[0, 0, i, 1])
            box = (detections[0, 0, i, 3:7]*np.array([W, H, W, H])).astype("int")
            
            # create dlib tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(box[0], box[1], box[2], box[3])
            tracker.start_track(img, rect)

            ls_trackers.append(tracker)
    
    return ls_trackers


def click_and_crop(event, x, y, flags, img):
    global door
    
    if init ==  0:
        if event == cv2.EVENT_LBUTTONDOWN:
            door = []
            door.append(x)
            door.append(y)
        elif event == cv2.EVENT_LBUTTONUP:
            # arange x, y to correct position
            door.insert(1 if x>door[0] else 0, x)
            door.insert(3 if y>door[2] else 2, y)
            
    if len(door) in range(2, 4):
        img_cp = img.copy()
        cv2.rectangle(img_cp, (door[0], door[1]), (x, y), (0, 255, 0), 2)
        cv2.imshow("output", img_cp)


def parse_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument("-p", "--prototxt", required=True,
                       help="path to Caffe 'deploy' prototxt file")
    parse.add_argument("-m", "--model", required=True,
                       help="path to Caffe pre-trained model")
    parse.add_argument("-c", "--confidence", type=float, default=0.4,
                       help="minimum probability to filter weak detections")
    parse.add_argument("-i", "--input", type=str,
                       help="path to input video file")
    parse.add_argument("-o", "--output", type=str,
                       help="path to optional output video file")
    parse.add_argument("-g", "--gap_size", type=int, default=10,
                       help="number of frames between 2 consecutive detections")
    return parse.parse_args()


def main():
    args = parse_arguments()
    classes =["background", "person"]
    # load caffe model
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    # take the first frame and wait for door init 
    cap = cv2.VideoCapture(args.input)
    _, img = cap.read()

    # resize image
    img = imutils.resize(img, width=400)
    H, W = img.shape[:2]

    # init 'output' window and mouse events
    cv2.namedWindow("output")
    cv2.setMouseCallback("output", click_and_crop, img)
    
    while True:
        # wait until door position is identify
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            # if 'r' is pressed, start program 
            global init
            init = 1
            break
    
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (W, H), True)

    # list of tracking objects detected by detector
    ls_trackers = []

    if len(door) == 4:
        # if door is identify correctly

        # create Counter object for counting people come in, and Tracker for tracking old-detected-people
        counter = Counter(door)
        my_tracker = Tracker()

        # varriable for counting frame elapsed
        elapsed_frame = -1
        while cap.isOpened():
            # open stream successfully
            _, img = cap.read()
            img = imutils.resize(img, width=400)
            
            elapsed_frame += 1
            
            # recored 40 frames for tracking objects inversely, [newest, ..., oldest]
            if elapsed_frame % 4 == 0:
                if len(counter.inv_frames) > 40:
                    counter.inv_frames.pop()
                counter.inv_frames.insert(0, img)

            rects = []

            # activate detector every args['gap-size'] frame(s)
            if elapsed_frame % args.gap_size == 0:
                ls_trackers = detect(net, img, args.confidence, W, H)
                
            else:
                # update new positions for old-objects
                for tracker in ls_trackers:
                    tracker.update(img)
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))

            # update new position for old object and return list of objects need to fing origin position
            inv_boxes = my_tracker.update(rects)
            
            # find the origin position
            counter.find_start_point(img, inv_boxes)

            for id, box in my_tracker.tracking_boxes.items():
                cv2.rectangle(img,
                            (box[0], box[1]), (box[2], box[3]),
                            (255, 255, 00), 2)

            cv2.putText(img, '#_in: '+str(counter.come_in),
                        (0, H-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)

            cv2.rectangle(img, 
                          (door[0], door[2]), (door[1], door[3]),
                          (0, 255, 0), 2)

            cv2.imshow('output', img)
            if args.output is not None:
                writer.write(img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
    
    if args.output is not None:
        writer.release()
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
    