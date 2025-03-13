
import os
import re
import xml.etree.ElementTree as ET
import time
import argparse

def main(args):
   DATA_PATH = args.data_path
   PROJ_NAME = args.project_name
   ITERATIONS = args.iterations
   RESULTS_PATH = args.results_path
   def post_process_result(str):
      def extract_alphanumeric(input_string):
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
   img_path = os.path.join(DATA_PATH, 'photos')
   xml_path = os.path.join(DATA_PATH, 'annotations.xml')
   real_values = []
   predictions = []

   for img in os.listdir(img_path):
      images.append(os.path.join(img_path, img))
   print("found images: " + str(len(images)))
   for img in images:
      real_values.append(get_license_plate_number(img, xml_path))
   print("found real values: " + str(len(real_values)))

   for i in range(int(ITERATIONS)):
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
      if not os.path.exists(os.path.join(f'{RESULTS_PATH}', 'Results')):
         os.makedirs(os.path.join(f'{RESULTS_PATH}', 'Results'))
      with open(os.path.join(f'{RESULTS_PATH}', 'Results', f'{PROJ_NAME}_results.txt'), 'a') as f:
         f.write(f"[{PROJ_NAME}] (Iteration {i + 1}): Accuracy: {result/len(images)}, Time: {end_time - start_time}\n")
         
if __name__ == '__main__':
   argparser = argparse.ArgumentParser()
   argparser.add_argument('-n', '--project_name', type=str, required=True)
   argparser.add_argument('-d', '--data_path', type=str, required=True)
   argparser.add_argument('-i', '--iterations', type=int, default=10)
   argparser.add_argument('-r', '--results_path', type=str, required=True)
   main(argparser.parse_args())