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

  cnn = load_model(r'C:\Users\X\source\repos\_AiP\license-plate-recognition-benchmark\Models\CNN\modelCNN.h5')
  # make prediction
  plate_location = cnn.predict(resized_test_image_list)

  car_image_copy = resized_test_image_list[0].copy()
  car_image_copy = car_image_copy * 255
  car_image_copy = car_image_copy.astype('uint8') # cast to uint8 so that we can plot with normal RGB color

  top_left = (int(plate_location[0][1] * 200), int(plate_location[0][3] * 200))
  bottom_right = (int(plate_location[0][0] * 200), int(plate_location[0][2] * 200))

  plate_location_origianl_scale = list()
  # display the car image with plate
  # cv2.rectangle(car_image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=2)
  # plt.imshow(cv2.cvtColor(car_image_copy, cv2.COLOR_BGR2RGB))


  # Przekształcamy lokalizację tablicy do oryginalnej skali
  plate_location_resized = plate_location[0] * 200
  test_image_size = test_image_orginal_size[0]
  height, width = test_image_size[0], test_image_size[1]

  original_xMax = plate_location_resized[0] * (width / IMAGE_RESIZE_X)
  original_xMin = plate_location_resized[1] * (width / IMAGE_RESIZE_X)
  original_yMax = plate_location_resized[2] * (height / IMAGE_RESIZE_Y)
  original_yMin = plate_location_resized[3] * (height / IMAGE_RESIZE_Y)

  # plt.figure(figsize=(30,10))

  # plt.subplot(1, len(orginal_test_image_list), 0+1)


  # copy the original image so that the original image stays unchanged
  image_copy = orginal_test_image_list[0].copy()

  # locate the corner of the plate
  bottom_right = (plate_location_origianl_scale[0][0] + 60, plate_location_origianl_scale[0][2] + 30)
  top_left = (plate_location_origianl_scale[0][1] - 60, plate_location_origianl_scale[0][3] - 30)

  # display plates
  cv2.rectangle(image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=10)
  plate_image = orginal_test_image_list[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

  hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
  lower_blue = np.array([100, 150, 50])
  upper_blue = np.array([140, 255, 255])

  blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
  plate_image[blue_mask > 0] = [255, 255, 255]

  height, width = plate_image.shape[:2]
  new_width, new_height = int(width // 2.2), int(height // 2.2)
  result = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_AREA)  

  result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
  _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  result = cv2.erode(result, np.ones((3, 3), np.uint8), iterations=1)
  result = cv2.dilate(result, np.ones((3, 3), np.uint8), iterations=1)
  
  plate_image = orginal_test_image_list[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0],:]
  plate_image_list.append(plate_image)

  reader = easyocr.Reader(['en','pl'])
  bounds = reader.readtext(result,contrast_ths=0.5, adjust_contrast=0.7)

  title_text = ''
  for text in bounds:
      title_text += text[1] + ' '
  return title_text

if __name__ == '__main__':
  import os
  import re
  import xml.etree.ElementTree as ET
  import time
  PROJ_NAME = 'CNN'
  def post_process_result(str):
    def extract_alphanumeric(input_string):
        # Use regex to find all alphanumeric characters (A-Z, a-z, 0-9)
        alphanumeric_characters = re.findall(r'[A-Za-z0-9]', input_string)
        result = ''.join(alphanumeric_characters)
        return result
    return extract_alphanumeric(str)
  def get_license_plate_number(img_path, xml_file_path):
    img_name = os.path.basename(img_path)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for image in root.findall('image'):
        if image.get('name') == img_name:
          box = image.find('box')
          if box is not None:
                attribute = box.find('attribute')
                if attribute is not None and attribute.get('name') == 'plate number':
                    return attribute.text
    raise Exception(f'No plate number found for image {img_name} in {xml_file_path}')
  
  images = []
  img_path = 'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\photos'
  xml_path = 'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\annotations.xml'
  real_values = []
  predictions = []
  iterations = 10
  
  for img in os.listdir(img_path):
      images.append(os.path.join(img_path, img))
  print("found images: " + str(len(images)))
  for img in images:
      real_values.append(get_license_plate_number(img, xml_path))
  print("found real values: " + str(len(real_values)))
  
  for i in range(iterations):
    result = 0
    start_time = time.time()
    for img in images:
        try:
              prediction = predict(img)
              predictions.append(prediction)
        except:
              predictions.append("error")
    end_time = time.time()
    for pred, rv in zip(predictions, real_values):
        if post_process_result(pred) == rv:
              result += 1
    print(f"Accuracy: {result/len(images)}, Time: {end_time - start_time}")
    with open(f'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\{PROJ_NAME}_results.txt', 'a') as f:
        f.write(f"[{PROJ_NAME}] (Iteration {i}): Accuracy: {result/len(images)}, Time: {end_time - start_time}\n")
