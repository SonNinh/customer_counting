from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict
import dlib
import cv2


class Tracker:
    def __init__(self, maxDisappeared=40, maxDistance=50):
        # max number of continuous frames which each object is not updated position
        self.maxDisappeared = maxDisappeared
        # dictionary for number of frames which each object is not updated position
        self.disappeared = OrderedDict()

        # max distance between 2 positions in 2 frames of 1 object 
        self.maxDistance = maxDistance

        # dictionary for bounding boxs of trackable object
        self.tracking_boxes = OrderedDict()

        # dictionary for centers of trackable object
        self.tracking_cens = OrderedDict()

        # the id of newset object
        self.last_id = 0


    def register(self, box, cen):
        '''
        add new objects to tracking dictionary
        '''
        self.tracking_boxes[self.last_id] = box
        self.tracking_cens[self.last_id] = cen
        self.disappeared[self.last_id] = 0
        self.last_id += 1


    def deregister(self, ID):
        '''
        remove object which is disappeared
        '''
        del self.tracking_boxes[ID]
        del self.tracking_cens[ID]
        del self.disappeared[ID]


    def update(self, new_objs):
        '''
        Update new positions for old objects and return list of objects need to find origin position
        new_objs = [[minX, minY, maxX, maxY], ...]
        '''

        new_cens = np.zeros((len(new_objs), 2), dtype="int")\
        # compute new centers of new_objs
        for i, (minX, minY, maxX, maxY) in enumerate(new_objs):
            cX = int((minX + maxX) / 2)
            cY = int((minY + maxY) / 2)
            new_cens[i] = (cX, cY)

        
        if len(self.tracking_boxes) == 0:
            # if tracker is not tracking any object,
            # put all new detected objects to tracking list
            for i, obj in enumerate(new_objs):
                self.register(obj, new_cens[i])
            return new_objs

        elif len(new_objs) == 0:
            # if there is nothing to match to old objects
            IDs = [id for id in self.disappeared.keys()]
            for id in IDs:
                self.disappeared[id] += 1
                if self.disappeared[id] > self.maxDisappeared:
                    self.deregister(id)
            return []

        else:
            # else,
            # compute distances of each pair of tracking_cens and new_detected_center
            obj_IDs = list(self.tracking_cens.keys())
            obj_cens = list(self.tracking_cens.values())
            D = dist.cdist(np.array(obj_cens), new_cens)

            # for each tracking_cens, find the nearest new_detected_center by value,
            # then sort the rerult by index
            rows = D.min(axis=1).argsort()

            # for each tracking_cens, find the nearest new_detected_center by index,
            # then sort the result arcording to rows
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            for row, col in zip(rows, cols):
                if col in used_cols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                
                # update new center and bouding box for tracking 
                ID = obj_IDs[row]
                self.tracking_cens[ID] = new_cens[col]
                self.tracking_boxes[ID] = new_objs[col]
                
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                # loop old objects which do not match any new objects
                for row in unused_rows:
                    ID = obj_IDs[row]
                    self.disappeared[ID] += 1
                    # delete objects whose 'disappeared' is greater then maximum
                    if self.disappeared[ID] > self.maxDisappeared:
                        self.deregister(ID)

            # find new objects which does not match any old object
            for col in unused_cols:
                self.register(new_objs[col], new_cens[col])

            return [new_objs[i] for i in unused_cols]


class Counter:
    def __init__(self, door_pos):
        self.inv_frames = list()
        self.come_in = 0
        self.door = door_pos

    def find_start_point(self, img, boxes):
        tracker = dlib.correlation_tracker()

        for box in boxes:
            rect = dlib.rectangle(box[0], box[1], box[2], box[3])
            tracker.start_track(img, rect)
            # current position of object
            cur_pos_y = (box[1]+box[3])/2

            # find the mean y-position which is under the door over n frames in the past
            num = 0
            sum_y = 0
            for frame in self.inv_frames:
                tracker.update(frame)
                pos = tracker.get_position()
                cen_X = int(pos.left()+pos.right())/2
                cen_Y = int(pos.top()+pos.bottom())/2

                if cen_Y > self.door[2] and cen_X > self.door[0] and cen_X < self.door[1]:
                    sum_y += cen_Y
                    num += 1

            
            if num != 0 and cur_pos_y+2 < sum_y/num:
                # if current position is above the door, the object come from outside
                self.come_in += 1
    
