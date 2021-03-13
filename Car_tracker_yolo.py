from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from shapely.geometry import Point, Polygon

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from color import color_cloth

#top left top right bottom left bottom right
src = np.float32(np.array(((1,120),(1279,120),(1,360),(1279,360))))
W_warp=600
H_warp=1600
dst = np.float32([[0, 0], [W_warp, 0], [0, H_warp], [W_warp, H_warp]])
matrix= cv2.getPerspectiveTransform(src, dst)
coords = [(1,120),(1,360),(1279,360),(1279,120),(1,120)]
poly = Polygon(coords)

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/traffic-cars.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/test_results.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

#counter = []
counter_frame=0
output_counter=0

d_min=0
min_dist_frame_counter=0
id1=int()
id2=int()
id1 =None
id2 =None
person_id=[]
last_person_id=[]
start_time={}
end_time={}
speed={}
while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    counter_frame=counter_frame+1
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)
    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)
    
    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)
    
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)
    people_counter=0
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        if class_name=='car' :
            #if ( Point((int((bbox[0]+bbox[2])/2) ,  int(bbox[3]))).within(poly)):
            if ( Point(int(bbox[2]) ,  int(bbox[3]) ).within(poly) ):
                #print(1,counter_frame,track.track_id,person_id)
                if (track.track_id not in person_id):
                    person_id.append(track.track_id)
                    start_time[track.track_id]=counter_frame

                    #print(2,counter_frame,track.track_id,person_id)
            if ( Point(int(bbox[2]) ,  int(bbox[3]) ).within(poly) ):
                pass
            else:
                if (track.track_id in person_id):
                    #cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 2)
                    if (track.track_id not in last_person_id):
                        last_person_id.append(track.track_id)
                        end_time[track.track_id]=counter_frame
                        #print(4,counter_frame,track.track_id,person_id)
                    else:
                        #cv2.putText(img,str(track.track_id),(int(bbox[0]+15), int(bbox[1]+25)), 0, 0.75,(255, 0, 0), 2)
                        speed[track.track_id]=(H_warp/100)/((end_time[track.track_id]-start_time[track.track_id])/vid_fps)
                        speed[track.track_id]=(speed[track.track_id]*3600)/1000
                        display=str(int(speed[track.track_id]))+' KPH'
                        cv2.putText(img,display,(int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                            (0, 0, 255), 2)
                        #print(track.track_id ,speed[track.track_id],start_time[track.track_id],end_time[track.track_id])
                        #print(5,counter_frame,track.track_id,person_id)
    print(counter_frame)
    
    fps = 1./(time.time()-t1)
    #cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    #cv2.putText(img, str(counter_frame), (0,30), 0, 1, (0,0,255), 2)
    cv2.line(img, tuple(src[0]), tuple(src[1]), (0, 255, 0), thickness=1)
    cv2.line(img, tuple(src[0]), tuple(src[2]), (0, 255, 0), thickness=1)
    cv2.line(img, tuple(src[3]), tuple(src[1]), (0, 255, 0), thickness=1)
    cv2.line(img, tuple(src[3]), tuple(src[2]), (0, 255, 0), thickness=1)
    out.write(img)
    
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()