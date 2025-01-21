import cv2
from vt_lpr_code import yolo3

def predict(image_path):

    cap = cv2.VideoCapture(image_path)
    ret, cv_img = cap.read()

    _, _, _, _, _, lp = yolo3(cv_img, "prediction_result.csv")

    return lp
