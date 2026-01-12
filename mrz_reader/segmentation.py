from ultralytics import YOLO
import cv2

class SegmentationNetwork:
    def __init__(self, model_path="weights/mrz_detector/mrz.pt"):
        self.model = YOLO(model_path)

    def predict(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        results = self.model(img)[0]

        detections = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections[label] = (x1, y1, x2, y2)

        return detections


class FaceDetection:
    

    def __init__(self, prototxt_path, caffemodel_path):
        
        self.faceNet = cv2.dnn.readNet(prototxt_path, caffemodel_path)

    def detect(self, image, confidence_input):
        
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_input:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                roi = img[startY:endY, startX:endX].copy()
                return roi, confidence
        return None, None
