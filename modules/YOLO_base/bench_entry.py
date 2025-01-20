import cv2
from vt_lpr_code import yolo3

def predict(image_path):

    cap = cv2.VideoCapture(image_path)
    ret, cv_img = cap.read()

    _, _, _, _, _, lp = yolo3(cv_img, "prediction_result.csv")

    return lp

if __name__ == '__main__':
    import os
    import re
    import xml.etree.ElementTree as ET
    import time
    PROJ_NAME = 'YOLO_base'
    def post_process_result(str):
        def extract_alphanumeric(input_string):
            # Use regex to find all alphanumeric characters (A-Z, a-z, 0-9)
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
    img_path = 'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\photos'
    xml_path = 'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\annotations.xml'
    real_values = []
    predictions = []
    iterations = 10

    for img in os.listdir(img_path):
        images.append(os.path.join(img_path, img))
    print("found images: " + str(len(images)))
    for img in images:
        real_values.append(get_license_plate_number(img, xml_path))
    print("found real values: " + str(len(real_values)))

    for i in range(iterations):
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
        with open(f'C:\\Users\\X\\source\\repos\\_AiP\\license-plate-recognition-benchmark\\data\\{PROJ_NAME}_results.txt', 'a') as f:
            f.write(f"[{PROJ_NAME}] (Iteration {i}): Accuracy: {result/len(images)}, Time: {end_time - start_time}\n")