import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


IMAGE_RESIZE_X = 200
IMAGE_RESIZE_Y = 200

car_folder_path = '/content/plates'

car_image_name_list = list()
for car_image in os.listdir(car_folder_path):
    full_path = os.path.join(car_folder_path, car_image)
    car_image_name_list.append(full_path)


plate_folder_path = '/content/annotations'

plate_name_list = list()
for plate_file in os.listdir(plate_folder_path):
    full_path = os.path.join(plate_folder_path, plate_file)
    plate_name_list.append(full_path)

car_image_name_list.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
plate_name_list.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

# read in car images and resize
resized_car_images_list = list()
for full_path in car_image_name_list:
    car_image = cv2.imread(full_path)
    car_resize_image = cv2.resize(car_image, (IMAGE_RESIZE_X, IMAGE_RESIZE_Y))
    resized_car_images_list.append(np.array(car_resize_image))

print('Read in {} resized car images'.format(len(resized_car_images_list)))

resized_plate_location_list = list()
for full_path in plate_name_list:
    xml_file = open(full_path, 'r')
    bs = BeautifulSoup(xml_file, "xml")

    width = int(bs.find('width').text)
    height = int(bs.find('height').text)

    xMax = float(bs.find('xmax').text) * (IMAGE_RESIZE_X / width)
    xMin = float(bs.find('xmin').text) * (IMAGE_RESIZE_X / width)
    yMax = float(bs.find('ymax').text) * (IMAGE_RESIZE_Y / height)
    yMin = float(bs.find('ymin').text) * (IMAGE_RESIZE_Y / height)
    resized_plate_location_list.append([int(xMax), int(xMin), int(yMax), int(yMin)])

print('Read in {} resized plate info'.format(len(plate_name_list)))

import xml.etree.ElementTree as ET

plate_location_list = []

# Assuming plate annotations are stored in XML files
for plate_file in plate_name_list:
    tree = ET.parse(plate_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        plate_location_list.append((xmin, ymin, xmax, ymax))


plt.figure(figsize=(16,8))
for i in range(8):
    plt.subplot(2, 4, i+1)

    # locate the plate location based on label
    top_left = (resized_plate_location_list[i][1], resized_plate_location_list[i][3])
    bottom_right = (resized_plate_location_list[i][0], resized_plate_location_list[i][2])

    # draw bounding box on the copy of resized car image so that we have original image to train with
    car_image_copy = resized_car_images_list[i].copy()
    cv2.rectangle(car_image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=2)
    plt.imshow(cv2.cvtColor(car_image_copy, cv2.COLOR_BGR2RGB))



  def splitTrainValTestSet():
    resized_car_images_list_np = np.array(resized_car_images_list)
    resized_plate_location_list_np = np.array(resized_plate_location_list)

    # normalization
    normalized_X = resized_car_images_list_np / 255
    normalized_y = resized_plate_location_list_np / 200

    # Split data into train and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        normalized_X, normalized_y, test_size=0.2, random_state=7
    )
    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=11
    )

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)
    print("Shape of y_test:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
def setupModel():
  cnn = Sequential()

  cnn.add(keras.layers.Conv2D(filters=16, kernel_size=5, input_shape=(IMAGE_RESIZE_X,IMAGE_RESIZE_Y,3), padding='same'))
  cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
  cnn.add(keras.layers.Dropout(0.1))
  cnn.add(keras.layers.Conv2D(filters=32, kernel_size=5, padding='same'))
  cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
  cnn.add(keras.layers.Conv2D(filters=64, kernel_size=5, padding='same'))
  cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
  cnn.add(keras.layers.Dropout(0.1))
  cnn.add(keras.layers.Conv2D(filters=32, kernel_size=5, padding='same'))
  cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
  cnn.add(Flatten())
  cnn.add(Dense(128, activation="relu"))
  cnn.add(Dense(64, activation="relu"))
  cnn.add(keras.layers.Dense(4, activation="sigmoid"))
  return cnn


import keras
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16

cnn = setupModel()
X_train, X_val, X_test, y_train, y_val, y_test = splitTrainValTestSet()
cnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
train = cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16, verbose=1)


# Test
scores = cnn.evaluate(X_test, y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))

accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training set accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test set accuracy')
plt.title('Scores')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.yticks(np.linspace(0, 1, 11))

plt.legend()
plt.show()


# read in car images and resize

test_image_folder = '/content/plates'
test_image_list = ['4.jpg', '5.jpg', '6.jpg']
resized_test_image_list = list()
orginal_test_image_list = list()
test_image_orginal_size = list()
for test_image in test_image_list:
    full_path = os.path.join(test_image_folder, test_image)
    car_image = cv2.imread(full_path)
    orginal_test_image_list.append(car_image)
    test_image_orginal_size.append(car_image.shape)
    car_resize_image = cv2.resize(car_image, (IMAGE_RESIZE_X, IMAGE_RESIZE_Y))
    resized_test_image_list.append(np.array(car_resize_image))

resized_test_image_list = np.array(resized_test_image_list)
resized_test_image_list = resized_test_image_list / 256

# make prediction
plate_location = cnn.predict(resized_test_image_list)
print(plate_location)

# plot the result
plt.figure(figsize=(16,8))
for i in range(3):
    plt.subplot(1, 3, i+1)

    car_image_copy = resized_test_image_list[i].copy()
    car_image_copy = car_image_copy * 255
    car_image_copy = car_image_copy.astype('uint8') # cast to uint8 so that we can plot with normal RGB color

    top_left = (int(plate_location[i][1] * 200), int(plate_location[i][3] * 200))
    bottom_right = (int(plate_location[i][0] * 200), int(plate_location[i][2] * 200))

    print(top_left)
    print(bottom_right)

    # display the car image with plate
    cv2.rectangle(car_image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=2)
    plt.imshow(cv2.cvtColor(car_image_copy, cv2.COLOR_BGR2RGB))



# get the location of the detected plate in original image size
plate_location_origianl_scale = list()
for i in range(len(test_image_orginal_size)):
  # this is the location before normalization
  plate_location_resized = plate_location[i] * 200
  test_image_size = test_image_orginal_size[i]
  height, width = test_image_size[0], test_image_size[1]

  original_xMax = plate_location_resized[0] * (width / IMAGE_RESIZE_X)
  original_xMin = plate_location_resized[1] * (width / IMAGE_RESIZE_X)
  original_yMax = plate_location_resized[2] * (height / IMAGE_RESIZE_Y)
  original_yMin = plate_location_resized[3] * (height / IMAGE_RESIZE_Y)
  plate_location_origianl_scale.append([int(original_xMax), int(original_xMin), int(original_yMax), int(original_yMin)])

# plot the detected plate with car in the orginal images
plt.figure(figsize=(30,10))
for i in range(len(orginal_test_image_list)):
  plt.subplot(1, len(orginal_test_image_list), i+1)

  # copy the original image so that the original image stays unchanged
  image_copy = orginal_test_image_list[i].copy()

  # locate the corner of the plate
  top_left = (plate_location_origianl_scale[i][1], plate_location_origianl_scale[i][3])
  bottom_right = (plate_location_origianl_scale[i][0], plate_location_origianl_scale[i][2])

  # display plates
  cv2.rectangle(image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=10)
  plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

plate_image_list = list()

# plot the detected plate with car in the orginal images
plt.figure(figsize=(30,10))
for i in range(len(orginal_test_image_list)):
  plt.subplot(1, len(orginal_test_image_list), i+1)

  # copy the original image so that the original image stays unchanged
  image_copy = orginal_test_image_list[i].copy()

  image_size = test_image_orginal_size[i]
  image_height = image_size[0]
  image_width = image_size[1]

  box_image_ratio_height = (plate_location_origianl_scale[i][2] - plate_location_origianl_scale[i][3]) / image_height
  box_image_ratio_width = (plate_location_origianl_scale[i][0] - plate_location_origianl_scale[i][1]) / image_width

  height_coef = 1 + ((1 / (np.log(box_image_ratio_height))**2) / 2)
  width_coef = 1 + ((1 / (np.log(box_image_ratio_width))**2) / 2)
  #print(height_coef, width_coef)

  # locate the corner of the plate
  top_left = (int(plate_location_origianl_scale[i][1] / width_coef), int(plate_location_origianl_scale[i][3] / height_coef))
  bottom_right = (int(plate_location_origianl_scale[i][0] * width_coef), int(plate_location_origianl_scale[i][2] * height_coef))

  # display plates
  cv2.rectangle(image_copy, top_left, bottom_right, color=(0, 255, 0), thickness=3)
  plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
  
  plate_image = orginal_test_image_list[i][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0],:]
  plate_image_list.append(plate_image)


#!pip install easyocr
import easyocr

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
