'''
python3 detector.py -p my_ssd/MobileNetSSD_deploy.prototxt -m my_ssd/mobilenet_iter_1800.caffemodel -i videos/input.mp4
'''

import cv2
import time
import glob
import numpy as np
import dlib
import argparse
import math as m

from my_tracker.tracker import Tracker, Counter


def IoU(box_1, box_2):
    overlap_box = [0, 0, 0, 0]

    overlap_box[0] = max(box_1[0], box_2[0])
    overlap_box[1] = max(box_1[1], box_2[1])
    overlap_box[2] = min(box_1[2], box_2[2])
    overlap_box[3] = min(box_1[3], box_2[3])

    
    if overlap_box[0] < overlap_box[2] and overlap_box[1] < overlap_box[3]:
        area_overlap = (overlap_box[2]-overlap_box[0]) * (overlap_box[3]-overlap_box[1])
        area_box_1 = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
        area_box_2 = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])
        iou = area_overlap / (area_box_1 + area_box_2 - area_overlap)
        print(iou)
        return iou

    return 0


def euclidean_dist(point_1, point_2):
    return m.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)


def is_existed(checked_box, ls_detected_objs):
    checked_center = ((checked_box[0]+checked_box[2])//2, (checked_box[1]+checked_box[3])//2)
    min_dist = 9999
    min_idx = -1
    for i, box in enumerate(ls_detected_objs):
        box_center = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        dist = euclidean_dist(checked_center, box_center)
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    if min_idx >= 0 and IoU(checked_box, ls_detected_objs[min_idx]) > 0.8:
        ls_detected_objs[min_idx][0] = min(checked_box[0], ls_detected_objs[min_idx][0])
        ls_detected_objs[min_idx][1] = min(checked_box[1], ls_detected_objs[min_idx][1])
        ls_detected_objs[min_idx][2] = max(checked_box[2], ls_detected_objs[min_idx][2])
        ls_detected_objs[min_idx][3] = max(checked_box[3], ls_detected_objs[min_idx][3])
        return True
    
    return False




def parse_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument("-p", "--prototxt", required=True,
                       help="path to Caffe 'deploy' prototxt file")
    parse.add_argument("-m", "--model", required=True,
                       help="path to Caffe pre-trained model")
    parse.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="minimum probability to filter weak detections")
    parse.add_argument("-i", "--input", type=str,
                       help="path to input video file")
    parse.add_argument("-g", "--gap_size", type=int, default=10,
                       help="number of frames between 2 consecutive detections")
    return parse.parse_args()


def main():
    args = parse_arguments()
    classes =["background", "person"]
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    cap = cv2.VideoCapture(args.input)
    door = [200, 280, 150]
    counter = Counter(door)
    my_tracker = Tracker()
    trackers = []

    elapsed_frame = 0
    # for path_img in glob.glob('test_images/*.png'):
    #     img = cv2.imread(path_img)
    ls_detected_objs = []
    while(cap.isOpened()):
        _, img = cap.read()
        H, W = img.shape[:2]
        img = cv2.resize(img,(W//5,H//5))
        H, W = H//5, W//5
        img_cp = img.copy()

        elapsed_frame += 1
        # activate detector every args['gap-size'] frame(s)
        if elapsed_frame % 2 == 0:
            if len(counter.inv_frames) > 30:
                counter.inv_frames.pop()
            counter.inv_frames.insert(0, img)
        
        rects = []
        if elapsed_frame % args.gap_size == 0:

            # list of objects detected by detector
            ls_detected_objs = []
            trackers = []

            blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args.confidence:
                    # class_id = int(detections[0, 0, i, 1])
                    box = (detections[0, 0, i, 3:7]*np.array([W, H, W, H])).astype("int")
                    cv2.rectangle(img,
                              (box[0], box[1]), (box[2], box[3]),
                              (255, 255, 00), 2)
                    # cv2.putText(img, 
                    #             str(confidence),
                    #             (box[0], box[1]+20),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (255,0,255), 1)
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
                    tracker.start_track(img, rect)

                    trackers.append(tracker)
        else:
            for tracker in trackers:
                tracker.update(img)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                cv2.rectangle(img,
                              (startX, startY), (endX, endY),
                              (255, 255, 00), 2)

                rects.append((startX, startY, endX, endY))

        inv_boxes = my_tracker.update(rects)
        print('len: ', len(inv_boxes))
        counter.find_start_point(img_cp, inv_boxes)
        print(counter.come_in)


        cv2.imshow('outpt', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
    