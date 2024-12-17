import xml.etree.ElementTree as ET
import importlib.util
import os
import time
import re
import wexpect
from wexpect.console_reader import ConsoleReaderPipe
import sys
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
   
   def delete_virtualenv(self, model_name):
      venv_path = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH, model_name, 'venv')
      if not os.path.exists(venv_path):
         return
      session = wexpect.spawn(f'cmd.exe')
      session.expect('>')
      session.sendline(f'cd {self.ROOT_PATH}')
      session.expect('>')
      session.sendline(f'rmdir /s /q {venv_path}')
      session.expect('>')
      session.close()
      
   def create_virtualenv(self, model_name):
      model_path = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH, model_name)
      venv_path = os.path.join(model_path, 'venv')
      
      if os.path.exists(venv_path):
         self.log(f'Virtualenv for {model_name} already exists', 1)
      python_version = PY_VERSIONS[model_name]
      python_path = PY_PATHS[python_version]
      session = wexpect.spawn('cmd.exe')
      session.expect('>')
      session.sendline(f'cd {model_path}')
      session.expect('>')
      session.sendline(f'virtualenv -p {python_path} {venv_path}')      
      session.expect(r'[a-zA-Z]:[\\\/]([^<>:"|?*\r\n]+[\\\/]?)*')
      session.close()
   
   def stream_session_output(self, session, t_out=90):
      try:
         while True:
               # Read the next line of output
               line = session.readline().strip()
               if not line:
                  continue  # Skip if the line is empty
               print('session output: ', line)  # Print the output in real-time
      except wexpect.EOF:
         print("Child process has finished.")
      except Exception as e:
         print(f"An error occurred: {e}")
      finally:
         session.close()  # Ensure the process is closed when finished
      
   def activate_virtualenv(self, model_name):
      model_path = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH, model_name)
      activation_script = os.path.join(model_path, 'venv', 'Scripts', 'activate')
      session = wexpect.spawn('powershell.exe')
      session.expect('>')
      print('output: ', session.before)
      session.sendline(f'cd {model_path}')
      print('output: ', session.before)
      session.expect('>')
      print('output: ', session.before)
      session.sendline(activation_script)
      print('output: ', session.before)
      session.expect('>', timeout=90)
      print('output: ', session.before)
      print(session)
      return session
      
   def deactivate_virtualenv(self, session):
      session.sendline('deactivate')
      session.expect(wexpect.EOF)
      
   def install_requirements(self, model_name, current_session):
      venv_path = os.path.join(self.ROOT_PATH, model_name, 'venv')
      requirements_file = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH, model_name, self.REQUIREMENTS_FILE)
      pip_path = os.path.join(venv_path, 'Scripts', 'pip')
      session = current_session
      session.sendline(f'pip install -r {requirements_file}')
      self.stream_session_output(session)
      session.expect(r"\(venv\) PS C:\\[^>]+", timeout=180)
      
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
   def copy_bench_runner(self, model_name):
      bench_runner_path = os.path.join(self.ROOT_PATH, 'Benchmark', 'run_helper.py')
      model_path = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH, model_name)
      dest_path = os.path.join(model_path, 'temp_runner.py')
      try:
         with open(bench_runner_path, 'r') as src_file:
            with open(dest_path, 'w') as dest_file:
               dest_file.write(src_file.read())
      except Exception as e:
         self.log(f'Failed to copy bench_runner.py to {model_name}: {e}', 2)
         
   def run_bench_runner(self, model_name, current_session):
      bench_runner_path = os.path.join(self.ROOT_PATH, self.MODELS_FOLDERS_PATH, model_name, self.BENCH_RUNNER_FILE)
      session = current_session
      args = f'--project_name {model_name} --data_path {os.path.join(self.ROOT_PATH, self.DATA_PATH)} --iterations {self.ITERATIONS}'
      session.expect('(venv)')
      session.sendline(f'python -m {bench_runner_path} {args}')
      session.expect(wexpect.EOF, timeout=self.ITERATIONS * 150)
      self.log(f'Finished running bench_runner.py for {model_name}.')
   
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
      for model_name in self.model_names:
         self.CURRENT_MODEL = model_name
         self.ensure_init_files(os.path.join(self.MODELS_FOLDERS_PATH, model_name))
         self.create_virtualenv(self.CURRENT_MODEL)
         current_session = self.activate_virtualenv(self.CURRENT_MODEL)
         self.install_requirements(self.CURRENT_MODEL, current_session)
         self.copy_bench_runner(self.CURRENT_MODEL)
         self.run_bench_runner(self.CURRENT_MODEL, current_session)

if __name__ == '__main__':
   benchmark = Benchmark()
   benchmark.run()