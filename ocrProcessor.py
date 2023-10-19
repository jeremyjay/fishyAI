import easyocr
import cv2 as cv
import numpy as np

class EasyOCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract_score(self, img, roi_coords):
        # x, y, w, h = roi_coords
        
        # # Extract the ROI
        # roi = img[y:y+h, x:x+w]
        
        # # Convert ROI to grayscale
        # gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # # Adaptive thresholding to get binary image
        # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        
        # # Dilation to join broken parts of characters
        # # kernel = np.ones((2, 2), np.uint8)
        # # dilated = cv.dilate(binary, kernel, iterations=1)
        # dilated = binary
        
        # # Resize (optional, try without this step first)
        # resized = cv.resize(dilated, (dilated.shape[1]*2, dilated.shape[0]*2), interpolation=cv.INTER_LINEAR)
        # dilated = resized
        # cv.imshow('Processed ROI', dilated)
        # cv.waitKey(1)
        
        # results = reader.readtext(dilated)
        
        # text = results[0][1] if results else ""
        
        # try:
        #     result = int(text)
        # except ValueError:
        #     print(f"Could not convert '{text}' to an integer")
        #     return 0
        
        # return result
        return 0

    def extract_death(self, img, roi_coords):
        x, y, w, h = roi_coords

        # Extract the ROI
        roi = img[y:y+h, x:x+w]

        # Convert ROI to grayscale
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # Adaptive thresholding to get binary image
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

        # Dilation to join broken parts of characters
        # kernel = np.ones((2, 2), np.uint8)
        # dilated = cv.dilate(binary, kernel, iterations=1)
        dilated = binary
        # # Resize (optional, try without this step first)
        # resized = cv.resize(dilated, (dilated.shape[1]*2, dilated.shape[0]*2), interpolation=cv.INTER_LINEAR)
        resized = dilated
        # cv.imshow('Processed ROI', resized)
        # cv.waitKey(1)

        results = self.reader.readtext(resized)

        text = results[0][1] if results else ""

        # Extract the first 4 letters and convert to lowercase
        text = text[:4].lower()

        print(f"text: '{text}'")

        # Check for specific words and return accordingly
        if text == "gulp" or text == "gvlp" or text == "play":
            return 0
        else:
            return 1




# import easyocr

# reader = easyocr.Reader(['en'])  # Initialize the EasyOCR reader for English

# def extract_score(img, roi_coords):
#     # x, y, w, h = roi_coords
    
#     # # Extract the ROI
#     # roi = img[y:y+h, x:x+w]
    
#     # # Convert ROI to grayscale
#     # gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    
#     # # Adaptive thresholding to get binary image
#     # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
#     # # Dilation to join broken parts of characters
#     # # kernel = np.ones((2, 2), np.uint8)
#     # # dilated = cv.dilate(binary, kernel, iterations=1)
#     # dilated = binary
    
#     # # Resize (optional, try without this step first)
#     # resized = cv.resize(dilated, (dilated.shape[1]*2, dilated.shape[0]*2), interpolation=cv.INTER_LINEAR)
#     # dilated = resized
#     # cv.imshow('Processed ROI', dilated)
#     # cv.waitKey(1)
    
#     # results = reader.readtext(dilated)
    
#     # text = results[0][1] if results else ""
    
#     # try:
#     #     result = int(text)
#     # except ValueError:
#     #     print(f"Could not convert '{text}' to an integer")
#     #     return 0
    
#     # return result
#     return 0


# def extract_death(img, roi_coords):
#     x, y, w, h = roi_coords
    
#     # Extract the ROI
#     roi = img[y:y+h, x:x+w]
    
#     # Convert ROI to grayscale
#     gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    
#     # Adaptive thresholding to get binary image
#     binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
#     # Dilation to join broken parts of characters
#     # kernel = np.ones((2, 2), np.uint8)
#     # dilated = cv.dilate(binary, kernel, iterations=1)
#     dilated = binary
    
#     cv.imshow('Processed ROI', dilated)
#     cv.waitKey(1)
    
#     results = reader.readtext(dilated)
    
#     text = results[0][1] if results else ""
    
#     # Extract the first 4 letters and convert to lowercase
#     text = text[:4].lower()

#     print(f"text: '{text}'")

#     # Check for specific words and return accordingly
#     if text == "gulp" or text == "play":
#         return 0
#     else:
#         return 1