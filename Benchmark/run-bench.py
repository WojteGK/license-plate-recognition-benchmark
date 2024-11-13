import cv2
import xml.etree.ElementTree as ET
import Models.example

DATA_PATH = 'data/'
def load_images(images_directory):
   images = []
   for i in range(1, 6):
      image = cv2.imread(images_directory + str(i) + '.jpg')
      images.append(image)
   return images

def get_license_plate_number(photo_number, xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    for image in root.findall('image'):
        if image.get('name') == f'{photo_number}.jpg':
            box = image.find('box')
            if box is not None:
                attribute = box.find('attribute')
                if attribute is not None and attribute.get('name') == 'plate number':
                    return attribute.text
    return None
 
def prepare_bench():
   images = load_images(f'{DATA_PATH}/images/')
   for img in images:
      Models.example.predict(img)
   

def run_bench():
   images = load_images(f'{DATA_PATH}/images/')
   
   for img in images:
      Models.predict(images[img])

if __name__ == '__main__':
   prepare_bench()
   run_bench()