def create_dir(path):
  try: 
      if not os.path.exists(path):
        os.makedirs(path) 
      else:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)  
  except OSError as error: 
      print ('An error occurred creating the directory in the following path: ',path,' : ',error)
        
        
def generate_extra_points_V2(path_imageRAW, path_imageSR):
    
  imagen = cv2.imread(path_imageSR)
  height, width, _ = imagen.shape

  
  left_x = int(width/4)
  right_x   = int(width/2)+int(width/4)
  up_y    = int(height/4)
  down_y     = int(height/2)+int(height/4)

  center_x = int(width/2)
  center_y = int(height/2)

  center_coordinates_arriba_izquierda = (left_x,up_y)
  center_coordinates_arriba_derecha   = (right_x,up_y)
  center_coordinates_abajo_izquierda  = (left_x,down_y)
  center_coordinates_abajo_derecha    = (right_x,down_y)
  center_coordinates_centro           = (center_x,center_y)


  imagen2 = cv2.imread(path_imageRAW)
  RECORTEY  = int(imagen2.shape[0]/2)
  RECORTEX  = int(imagen2.shape[1]/2)

  extra_coordinates = [center_coordinates_arriba_izquierda,
                        center_coordinates_arriba_derecha,
                        center_coordinates_abajo_izquierda,
                        center_coordinates_abajo_derecha,
                        center_coordinates_centro
                       ]

  result = []
  for point in extra_coordinates:
    a1 = point[0] #es la x
    a2 = point[1] #es la y
    
    result.append(((int(a1-RECORTEX)),(int(a2-RECORTEY)),(int(a1+RECORTEX)),(int(a2+RECORTEY)) ))

  return result

        
def know_dimensions(path):
  img = cv2.imread(path)
  height, width, channels = img.shape
  print(height, width, channels)


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def createSR(image_path, model_path, image_sr_path):
  IMAGE_ORIGINAL_PATH = image_path
  sr = dnn_superres.DnnSuperResImpl_create()
  image = cv2.imread(IMAGE_ORIGINAL_PATH)
  sr.readModel(model_path)
  sr.setModel("fsrcnn", 2)
  result = sr.upsample(image)
  cv2.imwrite(image_sr_path, result)
    
    
def download_model_SR(model_name, model_dir):
    base_url = 'https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/'
    model_file = base_url+ model_name +'.pb?raw=true'
    download_model = model_dir
    wget.download(model_file, download_model) 
    return download_model


def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + 
                                        model_file,
                                        untar=True)
    return str(model_dir)

def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)
