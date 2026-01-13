import cv2
import numpy as np
import easyocr

from mrz_reader.segmentation import SegmentationNetwork, FaceDetection
from mrz_reader.utils import *
from mrz_reader.mrz_parser import parse_mrz
from mrz_reader.validator import cross_validate


def instantiate_from_config_easyocr(config, reload=False):
    
    print("Initializing EasyOCR...")
    return get_obj_from_str("easyocr.Reader", reload)(**config)

def get_obj_from_str(string, reload=False):
    
    import importlib

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class MRZReader:
    
    def __init__(self, 
                 easy_ocr_params: dict,
                 facedetection_protxt: str = "./weights/face_detector/deploy.prototxt",
                 facedetection_caffemodel: str = "./weights/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                 segmentation_model: str = "./weights/mrz_detector/mrz.pt"):
       
        self.segmentation = SegmentationNetwork(segmentation_model)
        self.face_detection = FaceDetection(facedetection_protxt, facedetection_caffemodel)
        self.ocr_reader = instantiate_from_config_easyocr(easy_ocr_params)

    def predict(self, image, do_facedetect=False, facedetect_coef=0.1, preprocess_config=None):

        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # 1. Run YOLO detection
        detections = self.segmentation.predict(img)

        results = {}

        # 2. Crop each detected field
        for field, box in detections.items():
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]
            # 3. OCR
            text = self.recognize_text(crop, preprocess_config or {})
            results[field] = text

        # -------- MRZ PARSING & VALIDATION --------

        mrz1 = results.get("MRZ-1", "")
        mrz2 = results.get("MRZ-2", "")

        if mrz1 and mrz2:
            mrz_data = parse_mrz(mrz1, mrz2)
            validation = cross_validate(mrz_data, results)
        else:
            mrz_data = {}
            validation = {}


        # Optional face detection
        face = None
        if do_facedetect:
            face, face_coef = self.face_detection.detect(img, facedetect_coef)

            # ---- MRZ POST PROCESSING ----
        mrz_raw = results.get("mrz", "")

        if isinstance(mrz_raw, list):
            mrz_text = " ".join([t[1] for t in mrz_raw])
        else:
            mrz_text = mrz_raw

        mrz_lines = [l for l in mrz_text.split("\n") if len(l) > 30]

        if len(mrz_lines) >= 2:
            mrz_data = parse_mrz(mrz_lines[0], mrz_lines[1])
            validation = cross_validate(mrz_data, results)
        else:
            mrz_data = {}
            validation = {}

        return {
            "ocr_results": results,
            "mrz_data": mrz_data,
            "validation": validation,
            "face": face
        }
   
    def recognize_text(self, image, preprocess_config):
        
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image

        # Preprocessing steps
        if preprocess_config.get("do_preprocess", False):
            img = self._preprocess_image(img, preprocess_config)

        return self.ocr_reader.readtext(img)

    def _preprocess_image(self, img, preprocess_config):
        
        img = resize(img)

        if preprocess_config.get("skewness", False):
            img = self._correct_skew(img)

        if preprocess_config.get("delete_shadow", False):
            img = self._delete_shadow(img)

        if preprocess_config.get("clear_background", False):
            img = self._clear_background(img)

        # Further image processing
        img = self._apply_morphological_operations(img)
        img = self._apply_threshold(img)

        return img

    def _correct_skew(self, img):
        
        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(gray_img)
            rotated = rotate(img, angle, (0, 0, 0))
            return rotated
        except Exception as e:
            print(f"Skew correction failed: {e}")
            return img

    def _delete_shadow(self, img):
        
        try:
            return delete_shadow(img)
        except Exception as e:
            print(f"Shadow deletion failed: {e}")
            return img

    def _clear_background(self, img):
        
        try:
            return clear_background(img)
        except Exception as e:
            print(f"Background clearing failed: {e}")
            return img

    def _apply_morphological_operations(self, img):
        
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        return img

    def _apply_threshold(self, img):
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        v = np.uint8(cv2.normalize(v, v, 50, 255, cv2.NORM_MINMAX))
        _, thresh0 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)
        return cv2.bitwise_or(thresh0, thresh1)
