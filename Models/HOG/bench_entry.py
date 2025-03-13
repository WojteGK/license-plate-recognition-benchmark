import cv2
import pickle
from Classify import Classify
from Tools import resize
import pytesseract


# Ścieżka do modelu
MODEL_PATH = "/Users/mikolaj/license-plate-recognition-benchmark/Models/HOG/svm_hog.pickle"

# Załaduj model
with open(MODEL_PATH, 'rb') as f:
    classifier = pickle.load(f)


def predict(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")

    image_resized = resize(image, height=500)
    resized_height, resized_width, _ = image_resized.shape

    pred_dict, roi_coordinates = Classify().classify_new_instance(image_resized, classifier)


    if not pred_dict:
        return "Brak tablicy"

    best_roi_name, _ = max(pred_dict.items(), key=lambda x: x[1][1])

    if best_roi_name not in roi_coordinates:
        return "Brak współrzędnych"

    x, y, w, h = roi_coordinates[best_roi_name]
    pred_box = (x, y, x + w, y + h)

    image = cv2.imread(image_path)
    orig_height, orig_width, _ = image.shape

    scale_x = orig_width / resized_width
    scale_y = orig_height / resized_height

    pred_box_scaled = (
        int(pred_box[0] * scale_x),
        int(pred_box[1] * scale_y),
        int(pred_box[2] * scale_x),
        int(pred_box[3] * scale_y),
    )

    plate_crop = image[pred_box_scaled[1]:pred_box_scaled[3], pred_box_scaled[0]:pred_box_scaled[2]]


    plate_text = pytesseract.image_to_string(plate_crop, config="--psm 7")
    return plate_text