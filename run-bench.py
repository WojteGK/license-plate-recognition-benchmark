import cv2
import xml.etree.ElementTree as ET
import importlib
import os
import time
import re

DATA_PATH = 'data'
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
      
def import_models():
    models_path = os.listdir('Models')
    if not models_path:
        print(f'[Error]: No folders found in Models/')
        return []
    
    model_objects = []
    
    for folder in models_path:
        folder_path = os.path.join('Models', folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith('.py') or file == '__init__.py':
               continue
            module_name = file[:-3]
            module_path = f'Models.{folder}.{module_name}'
            try:
               module = importlib.import_module(module_path)
               model_objects.append(module)
            except Exception as e:
               print(f'[Error]: Failed to import {module_path}: {e}')
    
    return model_objects

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
 
def prepare_bench():
   models = import_models()
   print(f'[Info]: {len(models)} models found')
   images = load_images()
   try:
      for model in models:
         if not hasattr(model, 'predict'):
            print(f'[Error]: {model} does not have a predict method')
            continue
         for img_path in images:
            print(model.predict(img_path))
         
   except Exception as e:
      print(f'[Error]: error occurred while preparing stage in model {model}: {e}')

def print_results(model_name, status, results):
   print(f'[Model]: {model_name}\t [Status]: {status}\t [Results]: {results}\t')
   
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
   
def run_bench():
   models = import_models()
   print(f'[Info]: {len(models)} models found')
   if not models:
      print(f'[Error]: no models found')
      raise Exception('No models found')
   
   for model in models:
      try:
         good_results = 0
         images = load_images()
         start_time = time.time()
         for img_path in images:
            abs_path = os.path.abspath(img_path)
            img_name = os.path.basename(img_path).split('.')[0]
            print(f'[Info]: Predicting image {img_name} in model {model.__name__}')
            try:
               prediction = model.predict(abs_path)
               prediction = post_process_result(prediction)
               print(f'[Info]: Prediction: {prediction}, real value: {get_license_plate_number(abs_path, f'{DATA_PATH}/annotations.xml')}')
            except Exception as e:
               print(f'[Error]: error occurred while predicting image {img_path} in model {model.__name__}: {e}')
               continue                     
            
            if prediction == get_license_plate_number(abs_path, f'{DATA_PATH}/annotations.xml'):
               good_results += 1
               
         time_result = time.time() - start_time
         result_args = (model.__name__, 'success', f'good results: {good_results}/{len(images)} in {time_result} seconds')
         print_results(*result_args)
         save_results(*result_args)
         
      except Exception as e:
         print(f'[Error]: error occured: {e}')
         save_results(f'{model.__name__}', 'failed', f'error occurred: {e}')

      
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
   print('Preparing benchmark...')
   prepare_bench()
   print(f'Benchmark prepared in {time.time() - start_time} seconds')
   print('Running benchmark...')
   start_time = time.time()
   run_bench()
   print(f'Benchmark finished in {time.time() - start_time} seconds')
   # debug()
