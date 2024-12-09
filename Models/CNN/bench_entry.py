import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import easyocr

def predict(image_path):
  IMAGE_RESIZE_X = 200
  IMAGE_RESIZE_Y = 200

  test_image_orginal_size = list()
  orginal_test_image_list = list()
  resized_test_image_list = list()

  car_image = cv2.imread(image_path)
  orginal_test_image_list.append(car_image)
  test_image_orginal_size.append(car_image.shape)
  car_resize_image = cv2.resize(car_image, (IMAGE_RESIZE_X, IMAGE_RESIZE_Y))
  resized_test_image_list.append(np.array(car_resize_image))

  resized_test_image_list = np.array(resized_test_image_list)
  resized_test_image_list = resized_test_image_list / 256

  cnn = load_model('modelCNN.h5')
  # make prediction
  plate_location = cnn.predict(resized_test_image_list)
  print(plate_location)

  plt.figure(figsize=(16,8))

  plt.subplot(1, 3, 1)

  car_image_copy = resized_test_image_list[0].copy()
  car_image_copy = car_image_copy * 255
  car_image_copy = car_image_copy.astype('uint8') # cast to uint8 so that we can plot with normal RGB color

  top_left = (int(plate_location[0][1] * 200), int(plate_location[0][3] * 200))
  bottom_right = (int(plate_location[0][0] * 200), int(plate_location[0][2] * 200))

  print(top_left)
  print(bottom_right)

  plate_location_origianl_scale = list()
  # display the car image with plate
  cv2.rectangle(car_image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=2)
  plt.imshow(cv2.cvtColor(car_image_copy, cv2.COLOR_BGR2RGB))

  # this is the location before normalization
  plate_location_resized = plate_location[0] * 200
  test_image_size = test_image_orginal_size[0]
  height, width = test_image_size[0], test_image_size[1]

  original_xMax = plate_location_resized[0] * (width / IMAGE_RESIZE_X)
  original_xMin = plate_location_resized[1] * (width / IMAGE_RESIZE_X)
  original_yMax = plate_location_resized[2] * (height / IMAGE_RESIZE_Y)
  original_yMin = plate_location_resized[3] * (height / IMAGE_RESIZE_Y)
  plate_location_origianl_scale.append([int(original_xMax), int(original_xMin), int(original_yMax), int(original_yMin)])

  plt.figure(figsize=(30,10))

  plt.subplot(1, len(orginal_test_image_list), 0+1)

  # copy the original image so that the original image stays unchanged
  image_copy = orginal_test_image_list[0].copy()

  # locate the corner of the plate
  top_left = (plate_location_origianl_scale[0][1], plate_location_origianl_scale[0][3])
  bottom_right = (plate_location_origianl_scale[0][0], plate_location_origianl_scale[0][2])

  # display plates
  cv2.rectangle(image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=10)
  plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

  plate_image_list = list()
  plt.figure(figsize=(30,10))
  plt.subplot(1, len(orginal_test_image_list), 1)

  # copy the original image so that the original image stays unchanged
  image_copy = orginal_test_image_list[0].copy()

  image_size = test_image_orginal_size[0]
  image_height = image_size[0]
  image_width = image_size[1]

  box_image_ratio_height = (plate_location_origianl_scale[0][2] - plate_location_origianl_scale[0][3]) / image_height
  box_image_ratio_width = (plate_location_origianl_scale[0][0] - plate_location_origianl_scale[0][1]) / image_width

  height_coef = 1 + ((1 / (np.log(box_image_ratio_height))**2) / 2)
  width_coef = 1 + ((1 / (np.log(box_image_ratio_width))**2) / 2)
  #print(height_coef, width_coef)

  # locate the corner of the plate
  top_left = (int(plate_location_origianl_scale[0][1] / width_coef), int(plate_location_origianl_scale[0][3] / height_coef))
  bottom_right = (int(plate_location_origianl_scale[0][0] * width_coef), int(plate_location_origianl_scale[0][2] * height_coef))

  # display plates
  cv2.rectangle(image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=3)
  plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
  
  plate_image = orginal_test_image_list[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0],:]
  plate_image_list.append(plate_image)

  reader = easyocr.Reader(['en'])

  plt.figure(figsize=(30,10))
  for i, plate in enumerate(plate_image_list):
    plt.subplot(1, len(plate_image_list), i+1)
    plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))

    bounds = reader.readtext(plate)
    title_text = ''
    for text in bounds:
      title_text += text[1] + ' '
    plt.title('Detected Plate Number: ' + title_text, fontdict={'fontsize':20})
