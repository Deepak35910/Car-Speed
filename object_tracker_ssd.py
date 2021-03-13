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

src = np.float32(np.array(((1125,26),(1755,75),(87,500),(1150,716))))
W_warp=600
H_warp=1200
dst = np.float32([[0, 0], [W_warp, 0], [0, H_warp], [W_warp, H_warp]])
matrix= cv2.getPerspectiveTransform(src, dst)
coords = [(1125,26),(1755,75),(87,500),(1150,716)]
poly = Polygon(coords)

def warpfunction(p):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return([px,py])

#class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
#yolo = YoloV3(classes=len(class_names))
#yolo.load_weights('./weights/yolov3.tf')
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/test.mp4')

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
while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    h, w = img.shape[:2]    
    counter_frame=counter_frame+1
    #img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_in = tf.expand_dims(img_in, 0)
    #img_in = transform_images(img_in, 416)

    t1 = time.time()

    #boxes, scores, classes, nums = yolo.predict(img_in)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    boxes=detections[0,0,:,3:7]
    scores=detections[0,0,:,2]
    classes=detections[0,0,:,1]
    boxes = np.expand_dims(boxes, axis=0)
    scores = np.expand_dims(scores, axis=0)
    classes = np.expand_dims(classes, axis=0)
    print(boxes.shape,scores.shape,classes.shape)
    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    #converted_boxes = convert_boxes(img, boxes[0])
    #converted_boxes = boxes[0].tolist()* np.array([w, h, w, h])
    #converted_boxes = converted_boxes.astype('int') 
    #converted_boxes=converted_boxes[converted_boxes.sum(axis=1)>0].tolist()
    print('------------------just to check')
    print('before',boxes)
    boxes1=np.zeros(boxes.shape,dtype='float')
    boxes1[:,:,0]=(boxes[:,:,0]+boxes[:,:,2])/2
    boxes1[:,:,1]=(boxes[:,:,1]+boxes[:,:,3])/2
    boxes1[:,:,2]=boxes[:,:,2]-boxes[:,:,0]
    boxes1[:,:,3]=boxes[:,:,3]-boxes[:,:,1]
    print('after',boxes1)
    converted_boxes = convert_boxes(img, boxes[0])
    
    print(converted_boxes)
    features = encoder(img, converted_boxes)
    print(len(converted_boxes),features.shape)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    #boxs = np.array([d.tlwh for d in detections])
    #scores = np.array([d.confidence for d in detections])
    #classes = np.array([d.class_name for d in detections])
    #indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    #detections = [detections[i] for i in indices]
    print(len(detections))
    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)
    person_id=[]
    person_center=[]
    people_counter=0
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        
        if class_name=='person':
            
            #cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
            #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
            #            +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
            #            (255, 255, 255), 2)
            #cv2.circle(img, ( int((bbox[0]+bbox[2])/2) ,  int(bbox[3]) ), 5, (0,255,0), 5)
            if ( Point((int((bbox[0]+bbox[2])/2) ,  int(bbox[3]))).within(poly)):
                person_center.append(warpfunction( np.array((int((bbox[0]+bbox[2])/2),int(bbox[3])) )))
                person_id.append(track.track_id)
                people_counter=people_counter+1
    
    if counter_frame>2 and people_counter>0:
        def dis_calc(person_center,person_id):
            D = dist.cdist(person_center, person_center, metric="euclidean")
            D = np.where(D==0, 100000, D) 
            d_min1=int(np.min(D))
            id01,id02=np.unravel_index(D.argmin(), D.shape)
            #id01=id01+1
            #id02=id02+1
            id01=person_id[id01]
            id02=person_id[id02]
            return(d_min1,id01,id02)
        
        if d_min==0:
            d_min,id1,id2=dis_calc(person_center,person_id)
        
        if (d_min > 0 and d_min < 100):
            min_dist_frame_counter=min_dist_frame_counter+1
            counter_two=1
            
            for track in tracker.tracks:
                #class_name_two= track.get_class()
                #if class_name_two=='person':
                    if track.track_id == id1:
                        bbox_one = track.to_tlbr()
                        person_center_one=( int((bbox_one[0]+bbox_one[2])/2) ,  int(bbox_one[3]) )
                    if track.track_id == id2:
                        bbox_two = track.to_tlbr()
                        person_center_two=( int((bbox_two[0]+bbox_two[2])/2) ,  int(bbox_two[3]) )
                    counter_two=counter_two+1
            d_min = int(np.linalg.norm(np.array(person_center_one) - np.array(person_center_two)) )
            cv2.rectangle(img, (int(bbox_one[0]),int(bbox_one[1])), (int(bbox_one[2]),int(bbox_one[3])), color, 2)
            cv2.rectangle(img, (int(bbox_two[0]),int(bbox_two[1])), (int(bbox_two[2]),int(bbox_two[3])), color, 2)
            cv2.putText(img, str(id1), (int(bbox_one[0]), int(bbox_one[1]-10)), 0, 0.75,
                    (255, 0, 0), 2)
            cv2.putText(img, str(id2), (int(bbox_two[0]), int(bbox_two[1]-10)), 0, 0.75,
                    (255, 0, 0), 2)
            if (min_dist_frame_counter >= 90 and min_dist_frame_counter <= 120):
                if output_counter == 0:
                    img_pass1=img[int(bbox_one[1]):int(bbox_one[3]),int(bbox_one[0]):int(bbox_one[2])]
                    img_pass2=img[int(bbox_two[1]):int(bbox_two[3]),int(bbox_two[0]):int(bbox_two[2])]
                    color_label1=color_cloth(img_pass1)
                    color_label2=color_cloth(img_pass2)
                    color_label=str(color_label1)+','+str(color_label2)
                output_counter=output_counter+1
                cv2.circle(img, ( person_center_one ), 5, (0,255,0), 5)
                cv2.circle(img, ( person_center_two ), 5, (0,255,0), 5)
                cv2.putText(img, color_label, (int(bbox_one[0]), int(bbox_one[1]-10)), 0, 0.75,
                    (255, 0, 0), 2)
                cv2.putText(img, str(d_min), (int(bbox_one[0]-10), int(bbox_one[1]-20)), 0, 0.75,
                    (255, 0, 0), 2)
                
                #window_name = 'image'
                #cv2.imshow(window_name,img_pass1)
                #cv2.waitKey(0)      
                #print(img_pass1.shape)
                #window_name = 'image'
                #cv2.imshow(window_name,img_pass2)
                #cv2.waitKey(0)      
                #print(img_pass2.shape)
        else:
            d_min=0
            id1=None
            id2=None
            min_dist_frame_counter=0
            output_counter=0    
            
   

     #   center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
     #   pts[track.track_id].append(center)
     #
     #   for j in range(1, len(pts[track.track_id])):
     #       if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
     #           continue
     #       thickness = int(np.sqrt(64/float(j+1))*2)
     #       cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

     #   height, width, _ = img.shape
     #   cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
     #   cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)

     #   center_y = int(((bbox[1])+(bbox[3]))/2)

     #   if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
     #       if class_name == 'car' or class_name == 'truck':
     #           counter.append(int(track.track_id))
     #           current_count += 1

    #total_count = len(set(counter))
    #cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    #cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)
    print(counter_frame)
    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.line(img, tuple(src[0]), tuple(src[1]), (0, 255, 0), thickness=1)
    cv2.line(img, tuple(src[0]), tuple(src[2]), (0, 255, 0), thickness=1)
    cv2.line(img, tuple(src[3]), tuple(src[1]), (0, 255, 0), thickness=1)
    cv2.line(img, tuple(src[3]), tuple(src[2]), (0, 255, 0), thickness=1)
    #cv2.resizeWindow('output', 1024, 768)
    #cv2.imshow('output', img)
    out.write(img)
    

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()