import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pycocotools
import pylab
import re
import shutil
import tensorflow as tf
import time

from cv2 import dnn_superres
from dict2xml import dict2xml
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

from Cluster import *
from Utils import *

def filter_data(unfiltered_data, selected_classes, min_score):

    boxes, clases, scores = unfiltered_data
    boxes_filter = []
    classes_filter = []
    scores_filter = []

    for item in range(len(boxes)):
        if clases[item] in selected_classes:
            boxes_filter.append(boxes[item].tolist())
            scores_filter.append(scores[item])
            classes_filter.append(clases[item])

    boxes  = np.array(boxes_filter)
    clases = np.array(classes_filter)
    scores = np.array(scores_filter)

    true_boxes  = boxes[scores > min_score]
    true_scores = scores[scores > min_score]
    true_clases = clases[scores > min_score]
    
    return [true_boxes, true_clases, true_scores]


def make_simple_inference_SRSR(detect_fn,category_index,frame_path,x1,y1,width,height,Triples_Translated,selected_classes,min_score):
  image_np = load_image_into_numpy_array(frame_path)
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]
  
  detections = detect_fn(input_tensor)
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}

  detections['num_detections'] = num_detections
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  image_np_with_detections = image_np.copy()

  unfiltered_data = [detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']]
  true_boxes, true_clases, true_scores = filter_data(unfiltered_data, selected_classes, min_score)
  detections['detection_boxes'] = true_boxes
  detections['detection_classes'] = true_clases
  detections['detection_scores'] = true_scores
  
  Triple_Box = []
  Triple_Class = []
  Triple_Score = []

  for i in range(len(true_boxes)):
      box_detected = true_boxes[i]

      ymin11 = ((((box_detected[0]*height))+y1)/2)
      xmin11 = ((((box_detected[1]*width))+x1)/2)
      ymax11 = ((((box_detected[2]*height))+y1)/2)
      xmax11 = ((((box_detected[3]*width))+x1)/2)

      coordenadas_good = []    
      coordenadas_good.append((ymin11/height))
      coordenadas_good.append((xmin11/width))
      coordenadas_good.append((ymax11/height))
      coordenadas_good.append((xmax11/width))
          
      Triple_Box.append(coordenadas_good)
      Triple_Class.append(true_clases[i])
      Triple_Score.append(true_scores[i])
  
  Triples_Translated.append((np.array(Triple_Box),np.array(Triple_Class),np.array(Triple_Score)))
  

def make_inference_SRSR(detect_fn,category_index,image_path,image_name,image_save,selected_classes,min_score,MODEL_SR_DIR):
  
  selected_frame = np.array(Image.open(image_path))
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  number_files_raw = 0
  number_files_ours = 0

  detections = detect_fn(input_tensor)
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()} 

  detections['num_detections'] = num_detections
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  image_np_with_detections = image_np.copy()

  unfiltered_data = [detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']]
  true_boxes, true_clases, true_scores = filter_data(unfiltered_data, selected_classes, 0.2)

  img = cv2.imread(image_path)
  img__READ = cv2.imread(image_path)
  height, width, channels = img__READ.shape  
  createSR(image_path,MODEL_SR_DIR+'/FSRCNN_x2.pb',MODEL_SR_DIR+'/SR_Frame.png')
  im = Image.open(MODEL_SR_DIR+'/SR_Frame.png')
    
  create_dir(image_save+'/AUX/')

  Triple_List = []
  Nodes_Filter = []

  number_files_raw = len(true_boxes)
  
  new_width = width*2
  new_height = height*2 

  number_files_ours_aux = 0

  for i in range(len(true_boxes)):
      number_files_ours_aux = number_files_ours_aux + 1
      box_detected = true_boxes[i]
      
      ymin = int(box_detected[0]*height*2)
      xmin = int(box_detected[1]*width*2)
      ymax = int(box_detected[2]*height*2)
      xmax = int(box_detected[3]*width*2)
    
      ymin11 = (box_detected[0])
      xmin11 = (box_detected[1])
      ymax11 = (box_detected[2])
      xmax11 = (box_detected[3])

      coordenadas_good = []
      coordenadas_good.append(ymin11)
      coordenadas_good.append(xmin11)
      coordenadas_good.append(ymax11)
      coordenadas_good.append(xmax11)

      a1 = (xmin+xmax)//2
      a2 = (ymin+ymax)//2
      RECORTEX = int(width/2)
      RECORTEY = int(height/2)

      im_crop_outside = im.crop((a1-RECORTEX, a2-RECORTEY, a1+RECORTEX, a2+RECORTEY))
      nombre_sr_332 = image_save+"/AUX/{}.jpg".format(str(number_files_ours_aux))
      im_crop_outside.save(nombre_sr_332, quality=100)

      make_simple_inference_SRSR(detect_fn,category_index,nombre_sr_332,a1-RECORTEX,a2-RECORTEY,width,height,Triple_List,selected_classes,min_score)

  PATH_SR_V2  = MODEL_SR_DIR+'SR_Frame.png'
  PATH_RAW_V2 = image_path
  
  fixed_points = generate_extra_points_V2(PATH_RAW_V2, PATH_SR_V2)
  string_extra = 'EXTRA-'

  for id, point in enumerate(fixed_points):
        im_crop_outside = im.crop(point)
        nombre_sr_332 = image_save+"/AUX/{}.jpg".format(str(string_extra+str(id)))
        im_crop_outside.save(nombre_sr_332, quality=100)
        a1e = point[0]
        a2e = point[1]
        make_simple_inference_SRSR(detect_fn,category_index,nombre_sr_332,a1e,a2e,width,height,Triple_List,selected_classes,min_score)

  number_files_ours = number_files_ours_aux + 5

  Final_List_Boxes = []
  Final_List_Clases = []
  Final_List_Scores = []

  for i in range(len(true_boxes)):
    if true_scores[i] > min_score:
      box_detected = true_boxes[i]
        
      ymin11 = (box_detected[0])
      xmin11 = (box_detected[1])
      ymax11 = (box_detected[2])
      xmax11 = (box_detected[3])

      coordenadas_good = []
      coordenadas_good.append(ymin11)
      coordenadas_good.append(xmin11)
      coordenadas_good.append(ymax11)
      coordenadas_good.append(xmax11)
        
      Final_List_Boxes.append(coordenadas_good)
      Final_List_Clases.append(true_clases[i])
      Final_List_Scores.append(true_scores[i])
    
  Triple_List.append((np.array(Final_List_Boxes),np.array(Final_List_Clases),np.array(Final_List_Scores)))
  clusted_cliques = create_clusters(Triple_List, threshold =0.1)

  LISTA_BOXES_OK=[]
  LISTA_CLASES_OK=[]
  LISTA_SCORES_OK=[]
  
  for indexcluster in range(len(clusted_cliques)):
    tripla = clusted_cliques[indexcluster]
    lista_boxes_lista = tripla[0].tolist()
    lista_clases_lista = tripla[1].tolist()
    lista_scores_lista = tripla[2].tolist()
    for index2 in range(len(lista_boxes_lista)):
        LISTA_BOXES_OK.append(lista_boxes_lista[index2])
        LISTA_CLASES_OK.append(lista_clases_lista[index2])
        LISTA_SCORES_OK.append(lista_scores_lista[index2])
      
  detections['detection_boxes'] = np.array(LISTA_BOXES_OK)
  detections['detection_classes'] = np.array(LISTA_CLASES_OK)
  detections['detection_scores'] = np.array(LISTA_SCORES_OK)

  unfiltered_data = [detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']]
  true_boxes, true_clases, true_scores = filter_data(unfiltered_data, selected_classes, 0.2)
  
  detections['detection_boxes'] = np.array(true_boxes)
  detections['detection_classes'] = np.array(true_clases)
  detections['detection_scores'] = np.array(true_scores)

  LISTA_BOXES_OK=[]
  LISTA_CLASES_OK=[]
  LISTA_SCORES_OK=[]

  for indexcluster in range(len(clusted_cliques)):
    tripla = clusted_cliques[indexcluster]
    lista_boxes_lista = tripla[0].tolist()
    lista_clases_lista = tripla[1].tolist()
    lista_scores_lista = tripla[2].tolist()
    indice = np.argmax(lista_scores_lista)
    LISTA_BOXES_OK.append(lista_boxes_lista[indice])
    LISTA_CLASES_OK.append(lista_clases_lista[indice])
    LISTA_SCORES_OK.append(lista_scores_lista[indice])
          
  detections['detection_boxes'] = np.array(LISTA_BOXES_OK)
  detections['detection_classes'] = np.array(LISTA_CLASES_OK)
  detections['detection_scores'] = np.array(LISTA_SCORES_OK)

  unfiltered_data = [detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']]
  true_boxes, true_clases, true_scores = filter_data(unfiltered_data, selected_classes, 0.2)
    
  detections['detection_boxes'] = np.array(true_boxes)
  detections['detection_classes'] = np.array(true_clases)
  detections['detection_scores'] = np.array(true_scores)
     
  viz_utils.visualize_boxes_and_labels_on_image_array(
      selected_frame,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=1000,
      min_score_thresh=min_score,
      agnostic_mode=False,
      line_thickness=1)

  display(Image.fromarray(selected_frame))
  plt.axis('off')
  plt.imshow(selected_frame)
  
  img = cv2.imread(image_path)
  frame_with_detections = image_save+'/'+image_name
  plt.savefig(frame_with_detections,  dpi=300 ,bbox_inches='tight',pad_inches = 0)
  plt.clf()
    
  boxes = detections['detection_boxes']
  scores = detections['detection_scores'],
  clases_detected = detections['detection_classes']
  image = Image.open(image_path)
  width, height = image.size
  
  output = []
  output.append(boxes.tolist())
  output.append(scores[0])
  output.append(clases_detected) 
  output.append(width)
  output.append(height)
  output.append(clusted_cliques)
  output.append(number_files_raw)
  output.append(number_files_ours)
  return output


def make_inference_SRSRRAW(detect_fn,category_index,image_path,image_name,image_save,selected_classes, min_score):

  image_detections = np.array(Image.open(image_path))
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  detections = detect_fn(input_tensor)
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()} 

  detections['num_detections'] = num_detections
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  image_np_with_detections = image_np.copy()

  unfiltered_data = [detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']]
  true_boxes, true_clases, true_scores = filter_data(unfiltered_data, selected_classes, min_score)

  img = cv2.imread(image_path)
  img__READ = cv2.imread(image_path)
  height, width, channels = img__READ.shape  
    
  BOXES_TRANSLATED  = []
  CLASES_TRANSLATED = []
  SCORES_TRANSLATED = []
  
  for i in range(len(true_boxes)):
   
    box_detected = true_boxes[i]   
    ymin11 = (box_detected[0])
    xmin11 = (box_detected[1])
    ymax11 = (box_detected[2])
    xmax11 = (box_detected[3])

    coordinates_corrected = []
    coordinates_corrected.append(ymin11)
    coordinates_corrected.append(xmin11)
    coordinates_corrected.append(ymax11)
    coordinates_corrected.append(xmax11)
      
    BOXES_TRANSLATED.append(coordinates_corrected)
    CLASES_TRANSLATED.append(true_clases[i])
    SCORES_TRANSLATED.append(true_scores[i])
    
       
  detections['detection_boxes'] = np.array(BOXES_TRANSLATED)
  detections['detection_classes'] = np.array(CLASES_TRANSLATED)
  detections['detection_scores'] = np.array(SCORES_TRANSLATED)
  
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=1000,
      min_score_thresh=min_score,
      agnostic_mode=False,
      line_thickness=1)

  display(Image.fromarray(image_detections))
  plt.axis('off')
  plt.imshow(image_detections)
  
  img = cv2.imread(image_path)
  path_out_frame = image_save+'/'+image_name
  plt.savefig(path_out_frame,  dpi=300 ,bbox_inches='tight',pad_inches = 0)
  plt.clf()


  boxes = detections['detection_boxes']
  scores = detections['detection_scores'],
  clases_detected = detections['detection_classes']
  image = Image.open(image_path)
  width, height = image.size

  output = []
  output.append(boxes.tolist())
  output.append(scores[0])
  output.append(clases_detected) 
  output.append(width)
  output.append(height)
    
  return output
