import cv2
import xml.etree.ElementTree as ET
import importlib
import os
import time
import re

ENTRY_SCRIPT_NAME = 'bench_entry.py'
DATA_PATH = 'data'
MODELS_FOLDERS_PATH = 'Models'
DEBUG = True
def ensure_init_files(directory):
    for root, files in os.walk(directory):
        if '__init__.py' not in files:
            init_file_path = os.path.join(root, '__init__.py')
            try:
               with open(init_file_path, 'a'):
                  pass
               print(f'Created: {init_file_path}')
            except Exception as e:
               print(f'[Error]: Failed to create {init_file_path}: {e}')
               continue
            
def load_images():
   images_directory = os.path.join(DATA_PATH, 'photos')
   images = []
   try:
      for img_file in os.listdir(images_directory):         
         images.append(str(os.path.join(images_directory, str(img_file))))
      return images
   except FileNotFoundError:
      print(f'[Error]: {images_directory} not found')
      return []
   except Exception as e:
      print(f'[Error]: {e}')
      return []
def has_entry_script(path):
   entry_script_name = ENTRY_SCRIPT_NAME
   for file in os.listdir(path):
      if file == entry_script_name:
         return True
   return False

def has_specified_attr(module, attr = 'predict'):
   if not hasattr(module, attr):
      return False
   return True
      
def import_scripts():
   folder_path = MODELS_FOLDERS_PATH
   dirs = os.listdir(folder_path)
   folders = []
   for dir in dirs:
      if not os.path.isdir(dir):
         continue
      folders.append(dir)
   entry_scripts = []
   for folder in folders:
      try:
         path = os.path.join(folder_path, folder)
         ensure_init_files(path)
         if not has_entry_script(path):
            raise Exception(f'No entry script found in {path}')
         entry_script = importlib.import_module(os.path.join(path, ENTRY_SCRIPT_NAME))
         if not has_specified_attr(entry_script):
            raise Exception(f'No specified attribute found in {path}')
         entry_scripts.append(entry_script)
         
      except Exception as e:
         print(f'[Error]: {e}')
         continue
      if not entry_scripts:
         raise Exception('No entry scripts found!')
   return entry_scripts

def get_license_plate_number(img_path, xml_file_path):
   # Extract the image name from the path
   img_name = os.path.basename(img_path)
   
   # Parse the XML file
   tree = ET.parse(xml_file_path)
   root = tree.getroot()
   
   # Search for the image element with the matching name
   for image in root.findall('image'):
      if image.get('name') == img_name:
         # Find the box element and then the attribute element with the plate number
         box = image.find('box')
         if box is not None:
               attribute = box.find('attribute')
               if attribute is not None and attribute.get('name') == 'plate number':
                  return attribute.text
   raise Exception(f'No plate number found for image {img_name} in {xml_file_path}')
   
def save_results(model_name, status, results):
   timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
   if not os.path.exists('Benchmark/Results'):
      os.makedirs('Benchmark/Results')
   with open(f'Benchmark/Results/benchmark_results_{timestamp}.txt', 'a') as file:
      file.write(f'[{timestamp}]| [Model]: {model_name}\t [Status]: {status}\t [Results]: {results}\t')
def post_process_result(str):
   def extract_alphanumeric(input_string):
      # Use regex to find all alphanumeric characters (A-Z, a-z, 0-9)
      alphanumeric_characters = re.findall(r'[A-Za-z0-9]', input_string)
      # Join the list of characters into a single string
      result = ''.join(alphanumeric_characters)
      return result
   
   return extract_alphanumeric(str)
   
def run_bench(log = True):
   scripts = import_scripts()
   if log: print(f'[Info]: Running benchmark with {len(scripts)} scripts')
   images = load_images()
   if log:  print(f'[Info]: {len(images)} images loaded')
   
   for script in scripts:
      try:
         good_results = 0
         images = load_images()
         start_time = time.time()
         for img_path in images:
            abs_path = os.path.abspath(img_path)
            img_name = os.path.basename(img_path).split('.')[0]
            if log: print(f'[Info]: Predicting image {img_name} using {script.__name__}')
            try:
               prediction = script.predict(abs_path)
               prediction = post_process_result(prediction)
               if log: print(f'[Info]: Prediction: {prediction}, real value: {get_license_plate_number(abs_path, f'{DATA_PATH}/annotations.xml')}')
            except Exception as e:
               if log: print(f'[Error]:{e} while predicting image {img_path} in model {script}')
               continue
            
            if prediction == get_license_plate_number(abs_path, f'{DATA_PATH}/annotations.xml'):
               good_results += 1
               
         time_result = time.time() - start_time
         result_args = (script.__name__, 'success', f'good results: {good_results}/{len(images)} in {time_result} seconds')
         if log: print(f'[Info]: {script.__name__} finished with {good_results}/{len(images)} good results in {time_result} seconds')
         save_results(*result_args)
      except Exception as e:
         print(f'[Error]: error occured: {e}')
         save_results(f'{script.__name__}', 'failed', f'error occurred: {e}')

      
def debug():
   try:
      print(f'[Info]: Loading images...')
      images = load_images()
      print(f'[Info]: {len(images)} images loaded')
   except Exception as e:
      print(f'[Error]: {e}')
   try:
      print(f'[Info]: Importing models...')
      models = import_models()
      print(f'[Info]: {len(models)} models imported')
   except Exception as e:
      print(f'[Error]: {e}')
   try:
      print(f'[Info]: Running benchmark...')
      run_bench()
   except Exception as e:
      print(f'[Error]: {e}')
      
if __name__ == '__main__':
   start_time = time.time()
   print('Starting benchmark...')
   run_bench()
   print(f'Benchmark finished in {time.time() - start_time} seconds')
   # debug()
