import cv2
import xml.etree.ElementTree as ET
import importlib
import os
import time

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
   photo_name = img_path.split('/')[-1].split('.')[0]
   tree = ET.parse(xml_file_path)
   root = tree.getroot()
   
   for image in root.findall('image'):
      if image.get('name') == f'{photo_name}.jpg':
         box = image.find('box')
         if box is not None:
               attribute = box.find('attribute')
               if attribute is not None and attribute.get('name') == 'plate number':
                  return attribute.text
   return None
 
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
            img_name = os.path.basename(img_path).split('.')[0]
            print(f'[Info]: Predicting image {img_name} in model {model}')
            try:
               #TODO: make predict take absolute path so it will work
               prediction = model.predict()
            except Exception as e:
               print(f'[Error]: error occurred while predicting image {img_path} in model {model}: {e}')
               continue                     
            
            if prediction == get_license_plate_number(img_path, f'{DATA_PATH}/annotations.xml'):
               good_results += 1
               
         time_result = time.time() - start_time
         result_args = (model, 'success', f'{good_results/len(images)} in {time_result} seconds')
         print_results(*result_args)
         save_results(*result_args)
         
      except Exception as e:
         print(f'[Error]: error occured: {e}')
         save_results(f'{model}', 'failed', f'error occurred: {e}')

      
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
   # start_time = time.time()
   # print('Preparing benchmark...')
   # prepare_bench()
   # print(f'Benchmark prepared in {time.time() - start_time} seconds')
   # print('Running benchmark...')
   # start_time = time.time()
   # run_bench()
   # print(f'Benchmark finished in {time.time() - start_time} seconds')
   debug()
