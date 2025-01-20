import xml.etree.ElementTree as ET
import os
import time
import re
from config import PY_VERSIONS, PY_PATHS

class Benchmark:
   SELF_DIR = os.path.dirname(os.path.abspath(__file__))
   ROOT_PATH = os.path.abspath(os.path.join(SELF_DIR, '..'))
   ENTRY_SCRIPT_NAME = 'bench_entry.py'
   DATA_PATH = 'data'
   IMAGES_PATH = 'photos'
   MODELS_FOLDERS_PATH = os.path.join('Models') # delete TEST after debugging
   REQUIREMENTS_FILE = 'requirements.txt'
   ITERATIONS = 1
   BENCH_RUNNER_FILE = 'bench_runner.py'
   log_file = ''
   model_names = []
   
   def log(self, msg, type = 0):
      '''Creates a log with the given message. Type 0 is for info, 1 for warning, 2 for error'''
      if type == 0:
         print(f'[Info]: {msg}')
         self.log_file += f'[Info]: {msg}\n'
      elif type == 1:
         print(f'[Warning]: {msg}')
         self.log_file += f'[Warning]: {msg}\n'
      elif type == 2:
         print(f'[Error]: {msg}')
         self.log_file += f'[Error]: {msg}\n'
      
   def load_model_names(self):
      for folder in os.listdir(self.MODELS_FOLDERS_PATH):
         if folder.startswith('__'):
            continue
         if self.has_entry_script(os.path.join(self.MODELS_FOLDERS_PATH, folder)):
            self.model_names.append(folder)

   def ensure_init_files(self, directory):
      for root, dirs, files in os.walk(directory):
         init_file_path = os.path.join(root, '__init__.py')
         if not os.path.exists(init_file_path):
            try:
                  with open(init_file_path, 'a'):
                     pass
                  self.log(f'Created: {init_file_path}')
            except Exception as e:
                  self.log(f'[Error]: Failed to create {init_file_path}: {e}', type=2)
                  continue
      
   def load_images(self):
      images_directory = os.path.join(self.DATA_PATH, self.IMAGES_PATH)
      images = []
      try:
         for img_file in os.listdir(images_directory):         
            images.append(str(os.path.join(images_directory, str(img_file))))         
         self.images = images
      except FileNotFoundError:
         self.log(f'{images_directory} not found', 2)
         self.images = images
      except Exception as e:
         self.log(f'{e}', 2)
         self.images = images
           
   def has_entry_script(self, path):
      entry_script_name = self.ENTRY_SCRIPT_NAME
      for file in os.listdir(path):
         if file == entry_script_name:
            return True
      return False
   
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

   def save_log(self):
      timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
      logs_path = os.path.join(self.ROOT_PATH, 'Benchmark', 'Logs')
      if not os.path.exists(logs_path):
         os.makedirs(logs_path)
      with open(os.path.join(logs_path, f'bench_log_{timestamp}.txt'), 'a') as file:
         file.write(self.log_file)
   
   def post_process_result(self, str):
      def extract_alphanumeric(input_string):
         # Use regex to find all alphanumeric characters (A-Z, a-z, 0-9)
         alphanumeric_characters = re.findall(r'[A-Za-z0-9]', input_string)
         # Join the list of characters into a single string
         result = ''.join(alphanumeric_characters)
         return result
      
      return extract_alphanumeric(str)

   def run(self):
      self.load_model_names()

if __name__ == '__main__':
   benchmark = Benchmark()
   benchmark.run()