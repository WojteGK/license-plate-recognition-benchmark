import xml.etree.ElementTree as ET
import os
import time
import re
from config import PY_VERSIONS, PY_PATHS
import subprocess

class Benchmark:
   SELF_DIR = os.path.dirname(os.path.abspath(__file__))
   ROOT_PATH = os.path.abspath(os.path.join(SELF_DIR, '..'))
   ENTRY_SCRIPT_NAME = 'bench_entry.py'
   DATA_PATH = 'data'
   IMAGES_PATH = 'photos'
   MODULE_FOLDERS_PATH = 'modules'
   REQUIREMENTS_FILE = 'requirements.txt'
   log_file = ''
   module_names = []
   
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
      
   def load_module_names(self):
      for folder in os.listdir(self.MODULE_FOLDERS_PATH):
         if folder.startswith('__'):
            continue
         if self.has_entry_script(os.path.join(self.MODULE_FOLDERS_PATH, folder)):
            self.module_names.append(folder)

   def get_module_python_version(self, module_name):
      if module_name in PY_VERSIONS:
         return PY_VERSIONS[module_name]
      else:
         raise Exception(f'Python version not found for module {module_name} in config.py')
      
   def setup_venv(self, module_name):
      python_version = self.get_module_python_version(module_name)
      python_path = PY_PATHS[python_version]
      module_path = os.path.join(self.MODULE_FOLDERS_PATH, module_name)
      venv_path = os.path.join(module_path, 'venv')
      python_exe = os.path.join(venv_path, "Scripts", "python.exe")
      if not os.path.exists(venv_path):
         try:
            subprocess.run([python_path, '-m', 'venv', venv_path], check=True)
            self.log(f'Venv created for {module_name}')
         except subprocess.CalledProcessError as e:
            self.log(f'Failed to create venv for {module_name}: {e}', type=2)
      else:
         self.log(f'Venv already exists for {module_name}')
      requirements_path = os.path.join(module_path, self.REQUIREMENTS_FILE)
      if os.path.exists(requirements_path):
         try:
            subprocess.run([python_exe, '-m', 'pip', 'install', '-r', requirements_path], check=True)
            self.log(f'Requirements installed for {module_name}')
         except subprocess.CalledProcessError as e:
            self.log(f'Failed to install requirements for {module_name}: {e}', type=2)
           
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

   def generate_temp_runner(self, module_name):
      helper_path = os.path.join(self.SELF_DIR, 'run_helper.py')
      bench_entry_path = os.path.join(self.MODULE_FOLDERS_PATH, module_name, self.ENTRY_SCRIPT_NAME)
      output_path = os.path.join(self.MODULE_FOLDERS_PATH, module_name, 'temp_runner.py')
      with open(helper_path, 'r') as helper_file:
         helper_content = helper_file.readlines()

      with open(bench_entry_path, 'r') as bench_file:
         bench_content = bench_file.readlines()
         
      if os.path.exists(output_path):
         os.remove(output_path)
         
      with open(output_path, 'w') as output_file:
         output_file.writelines(bench_content)
         output_file.writelines(helper_content)
      self.log(f'Generated temp runner for {module_name}')
   
   def setup_module(self, module_name):
      self.log(f'Preparing {module_name}...')
      self.setup_venv(module_name)
      self.generate_temp_runner(module_name)
   
   def run_bench_module(self, module_name, iterations):
      module_path = os.path.join(self.ROOT_PATH,self.MODULE_FOLDERS_PATH, module_name)
      print('module_path: ' + module_path) # problems with paths
      venv_path = os.path.join(module_path, 'venv')
      python_exe = os.path.join(venv_path, "Scripts", "python.exe")
      temp_runner_path = os.path.join(module_path, 'temp_runner.py')
      try:
         self.log(f'Running benchmark for {module_name}...')
         print(os.getcwd())
         subprocess.run([python_exe, temp_runner_path, 
                        '-n', module_name,
                        '-d', self.DATA_PATH,
                        '-i', str(iterations),
                        '-r', self.ROOT_PATH,
                        ], check=True, cwd=os.path.join(self.ROOT_PATH, self.MODULE_FOLDERS_PATH, module_name)) # problems with paths (cwd)
         print(os.getcwd())
      except subprocess.CalledProcessError as e:
         self.log(f'Failed to run benchmark for {module_name}: {e}', type=2)
         print(f'Failed to run benchmark for {module_name}: {e}')
      
   def run(self):
      self.load_module_names()
      print(f'Modules found: {self.module_names}')
      self.load_images()
      print(f'Images found: {len(self.images)}')
      for module_name in self.module_names:
         try:
            self.setup_module(module_name)
            self.run_bench_module(module_name, 1)
         except Exception as e:
            self.log(f'Failed to run benchmark for {module_name}: {e}', type=2)

if __name__ == '__main__':
   benchmark = Benchmark()
   benchmark.run()