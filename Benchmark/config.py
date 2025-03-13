import os

# dict with python versions for submodules.
# *****************************************
# Please note that each submodule name 
# should be the same as the name of 
# the folder in the Modules directory
PY_VERSIONS = {'Classic_methods': '3.12.4', 
               'CNN': '3.10', 
               'YOLO': '3.6.8', 
               'YOLO_base': '3.6.8'}

# Adjust paths if needed (default is for Windows)
PY_PATHS = {'3.12.4': f'C:\\Users\\{os.getenv('username')}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe', 
            '3.10': f'C:\\Users\\{os.getenv('username')}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe', 
            '3.6.8': f'C:\\Users\\{os.getenv('username')}\\AppData\\Local\\Programs\\Python\\Python36\\python.exe'}


