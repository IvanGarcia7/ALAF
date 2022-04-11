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


def make_simple_inference_SRSR(frame_path,x1,y1,width,height,Triples_Translated,selected_classes,factor,min_score):
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
      coordenadas_good.append((ymin11/height)*factor)
      coordenadas_good.append((xmin11/width)*factor)
      coordenadas_good.append((ymax11/height)*factor)
      coordenadas_good.append((xmax11/width)*factor)
          
      Triple_Box.append(coordenadas_good)
      Triple_Class.append(true_clases[i])
      Triple_Score.append(true_scores[i])
  
  Triples_Translated.append((np.array(Triple_Box),np.array(Triple_Class),np.array(Triple_Score)))
  

def make_inference_SRSR(image_path,image_name,image_save,selected_classes,factor,FACTOR2,min_score,MODEL_SR_DIR):
  
  selected_frame = np.array(Image.open(image_path))
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
  true_boxes, true_clases, true_scores = filter_data(unfiltered_data, selected_classes, 0.2)

  img = cv2.imread(image_path)
  img__READ = cv2.imread(image_path)
  height, width, channels = img__READ.shape  
  createSR(image_path,MODEL_SR_DIR+'/FSRCNN_x2.pb',MODEL_SR_DIR+'/SR_Frame.png')
  im = Image.open(MODEL_SR_DIR+'/SR_Frame.png')
    
  create_dir(image_save+'/AUX/')

  Triple_List = []
  Nodes_Filter = []

  for i in range(len(true_boxes)):
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
    
      coordenada1_y = ((box_detected[0]+box_detected[2])/2)*height
      coordenada2_x = ((box_detected[1]+box_detected[3])/2)*width
    
      Nodes_Filter.append(Node(str(i),(coordenada2_x,coordenada1_y),(int((xmin+xmax)/2),int((ymin+ymax)/2)),(xmin11*width,xmax11*width,ymin11*height,ymax11*height)))
      
  new_width = width*FACTOR2
  new_height = height*FACTOR2
  adj_matrix2 = []
  adj_matrix_SUPLE = []
  all_nodes = []

  for i in range(len(Nodes_Filter)):

     adj_matrix_SUPLE = []
     First_Node_Selected = Nodes_Filter[i]
     lista_vecinos = []

     a1 = First_Node_Selected.coordenadas[0]
     a2 = First_Node_Selected.coordenadas[1]

     RECORTEX = int(new_width/2)
     RECORTEY = int(new_height/2)
    
     xmin = a1-RECORTEX
     ymin = a2-RECORTEY
     xmax = a1+RECORTEX
     ymax = a2+RECORTEY
      
     for o in range(len(Nodes_Filter)):
       if i != o:
         Second_Node_Selected = Nodes_Filter[o]
         
         xminb = Second_Node_Selected.x1x2y1y2[0]
         xmaxb = Second_Node_Selected.x1x2y1y2[1]
         yminb = Second_Node_Selected.x1x2y1y2[2]
         ymaxb = Second_Node_Selected.x1x2y1y2[3]
        
         if (xminb >= xmin and xmaxb <= xmax and yminb >= ymin and ymaxb <= ymax):
           Nodes_Filter[i].neighbors.append(Nodes_Filter[o]) 
           adj_matrix_SUPLE.append(1)
         else:
            adj_matrix_SUPLE.append(0)
       else:
        adj_matrix_SUPLE.append(0)
     adj_matrix2.append(adj_matrix_SUPLE)          
  
  for i in range(len(Nodes_Filter)):
    all_nodes.append(Nodes_Filter[i])
    
  total_cliques = []
  NumNodes = {
    i: set(num for num, j in enumerate(row) if j)
    for i, row in enumerate(adj_matrix2)
  }
  P2 = NumNodes.keys()
  
  optimized_cliques_list = list(Optimizer_V2(bron_kerbosch(adj_matrix2, pivot=True),3))
  new_width = width*factor*2
  new_height = height*factor*2 

  for ix in optimized_cliques_list:

    coordinate_selected_list = []
    for ab in ix:
     coordenadaxyc =all_nodes[ab].coordenadas2
     coordinate_selected_list.append(coordenadaxyc)
    
    if len(coordinate_selected_list) > 0:
        centroide = centroid1(coordinate_selected_list)
        a1 = centroide[0]
        a2 = centroide[1]
        RECORTEX = int(new_width/2)
        RECORTEY = int(new_height/2)
        
        im_crop_outside = im.crop((a1-RECORTEX, a2-RECORTEY, a1+RECORTEX, a2+RECORTEY))
        nombre_sr_332 = image_save+"/AUX/{}.jpg".format(str(ix))
        im_crop_outside.save(nombre_sr_332, quality=100)

        make_simple_inference_SRSR(nombre_sr_332,a1-RECORTEX,a2-RECORTEY,new_width,new_height,Triple_List,selected_classes,factor*2,min_score)

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
        make_simple_inference_SRSR(nombre_sr_332,a1e,a2e,new_width,new_height,Triple_List,selected_classes,factor*2,min_score)

    
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
  
  return output


def make_inference_SRSRRAW(image_path,image_name,image_save,selected_classes, min_score):

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
    
  return salida