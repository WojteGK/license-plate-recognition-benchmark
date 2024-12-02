import xml.etree.ElementTree as ET
import importlib.util
import os
import time
import re

class Benchmark:
   SELF_DIR = os.path.dirname(os.path.abspath(__file__))
   ROOT_PATH = os.path.abspath(os.path.join(SELF_DIR, '..'))
   ENTRY_SCRIPT_NAME = 'bench_entry.py'
   DATA_PATH = 'data'
   IMAGES_PATH = 'photos'
   MODELS_FOLDERS_PATH = 'Models'
   SPECIFIED_ATTRIBUTE = 'predict'
   
   DEBUG = True
   
   images = []
   modules = []
   # TODO: create venv dynamically in wsl; https://chatgpt.com/share/674c58e0-ecf0-800c-a6fe-75515fccd2b5
   def ensure_init_files(self, directory):
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
   
   def load_images(self):
      images_directory = os.path.join(self.DATA_PATH, self.IMAGES_PATH)
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
   
   def has_entry_script(self, path):
      entry_script_name = self.ENTRY_SCRIPT_NAME
      for file in os.listdir(path):
         if file == entry_script_name:
            return True
      return False
   
   
   def has_specified_attr(self, module, attr = SPECIFIED_ATTRIBUTE):
      if not hasattr(module, attr):
         return False
      return True

   def import_scripts(self):
      folder_with_models = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH)
      script_folders = [] 
      entry_scripts = [] # as list of modules
      print(f'[Info]: Importing scripts from {folder_with_models}')
      
      for dir in os.listdir(folder_with_models):
         if dir.startswith('__'):
            continue
         print(f'[Info]: found "{dir}"')
         script_folders.append(dir)
      print(f'[Info]: Found {len(script_folders)} folders with scripts: {script_folders}')
      
      for folder in script_folders:
         try:
            if not self.has_entry_script(os.path.join(folder_with_models, folder)):
               raise Exception(f'No {self.ENTRY_SCRIPT_NAME} found in {folder}')
            print(f'[Info]: Importing {self.ENTRY_SCRIPT_NAME} from {folder}')
            script_path = os.path.join(folder_with_models, folder, self.ENTRY_SCRIPT_NAME)         
            module_name = f"{script_path}_{self.ENTRY_SCRIPT_NAME[:-3]}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not self.has_specified_attr(module):
               raise Exception(f'No {self.SPECIFIED_ATTRIBUTE} attribute found in {module_name}')
            entry_scripts.append(module)            
         
         except Exception as e:
            print(f'[Error]: {e}')
            continue
         
      if not entry_scripts:
         raise Exception('No entry scripts found!')
      return entry_scripts
   
   def get_license_plate_number(self, img_path, xml_file_path):
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

   def save_results(self, model_name, status, results):
      timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
      if not os.path.exists('Benchmark/Results'):
         os.makedirs('Benchmark/Results')
      with open(f'Benchmark/Results/benchmark_results_{timestamp}.txt', 'a') as file:
         file.write(f'[{timestamp}]| [Model]: {model_name}\t [Status]: {status}\t [Results]: {results}\t')

   def post_process_result(self, str):
      def extract_alphanumeric(input_string):
         # Use regex to find all alphanumeric characters (A-Z, a-z, 0-9)
         alphanumeric_characters = re.findall(r'[A-Za-z0-9]', input_string)
         # Join the list of characters into a single string
         result = ''.join(alphanumeric_characters)
         return result
      
      return extract_alphanumeric(str)

   def run(self, log = True):
      scripts = self.import_scripts()
      if log: print(f'[Info]: Running benchmark with {len(scripts)} scripts')
      images = self.load_images()
      if log:  print(f'[Info]: {len(images)} images loaded')
      
      for script in scripts:
         try:
            good_results = 0
            images = self.load_images()
            start_time = time.time()
            for img_path in images:
               abs_path = os.path.abspath(img_path)
               img_name = os.path.basename(img_path).split('.')[0]
               if log: print(f'[Info]: Predicting image {img_name} using {script.__name__}')
               try:
                  prediction = script.predict(abs_path)
                  prediction = self.post_process_result(prediction)
                  if log: print(f'[Info]: Prediction: {prediction}, real value: {self.get_license_plate_number(abs_path, f'{self.DATA_PATH}/annotations.xml')}')
               except Exception as e:
                  if log: print(f'[Error]:{e} while predicting image {img_path} in model {script}')
                  continue
               
               if prediction == self.get_license_plate_number(abs_path, f'{self.DATA_PATH}/annotations.xml'):
                  good_results += 1
                  
            time_result = time.time() - start_time
            result_args = (script.__name__, 'success', f'good results: {good_results}/{len(images)} in {time_result} seconds')
            if log: print(f'[Info]: {script.__name__} finished with {good_results}/{len(images)} good results in {time_result} seconds')
            self.save_results(*result_args)
         
         except Exception as e:
            print(f'[Error]: error occured: {e}')
            self.save_results(f'{script.__name__}', 'failed', f'error occurred: {e}')
