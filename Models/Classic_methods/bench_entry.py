import cv2
import pytesseract
import numpy as np

# Predict function wrapping the model code
def predict(image_path): # image path as a parameter
   
   image = cv2.imread(image_path) # Load the image into proper format
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(gray, 50, 150)
   contours, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   image_copy = image.copy()
   _ = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 2)
   contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

   image_reduced = edges.copy()
   _ = cv2.drawContours(image_reduced, contours, -1, (255, 0, 0), 2)
   for contour in contours:
      perimeter = cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

      if len(approx) == 4:
         license_plate = approx
         break

   mask = np.zeros(gray.shape, np.uint8)  
   cv2.drawContours(mask, [license_plate], 0, 255, -1)
   plate_image = cv2.bitwise_and(image, image, mask=mask)
   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
   plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
   plate_number = pytesseract.image_to_string(plate_image_gray)
   
   return plate_number # Return the prediction
