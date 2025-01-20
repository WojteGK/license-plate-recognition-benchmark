import cv2
import pytesseract
import numpy as np

# Predict function wrapping the model code
def predict(image_path): # image path as a parameter
   
   image = cv2.imread(image_path) # Load the image into proper format
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(gray, 50, 150)
   contours, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   image_copy = image.copy()
   _ = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 2)
   contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

   image_reduced = edges.copy()
   _ = cv2.drawContours(image_reduced, contours, -1, (255, 0, 0), 2)
   for contour in contours:
      perimeter = cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

      if len(approx) == 4:
         license_plate = approx
         break

   mask = np.zeros(gray.shape, np.uint8)  
   cv2.drawContours(mask, [license_plate], 0, 255, -1)
   plate_image = cv2.bitwise_and(image, image, mask=mask)
   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
   plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
   plate_number = pytesseract.image_to_string(plate_image_gray)
   
   return plate_number # Return the prediction

if __name__ == '__main__':
   import os
   import re
   import xml.etree.ElementTree as ET
   import time
   PROJ_NAME = 'Classic_methods'
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
   img_path = 'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\photos' # Path to the images 
   xml_path = 'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\annotations.xml' # Path to the XML file
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