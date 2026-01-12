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
    """
    A class to perform face detection using a Caffe model.

    Attributes:
    -----------
    faceNet : cv2.dnn_Net
        The loaded Caffe model for face detection.

    Methods:
    --------
    detect(image, confidence_input)
        Detects a face in the image and returns the region of interest (ROI).
    """

    def __init__(self, prototxt_path, caffemodel_path):
        """
        Initializes the FaceDetection with the given Caffe model files.

        Parameters:
        -----------
        prototxt_path : str
            Path to the Caffe model's deploy.prototxt file.
        caffemodel_path : str
            Path to the Caffe model's .caffemodel file.
        """
        self.faceNet = cv2.dnn.readNet(prototxt_path, caffemodel_path)

    def detect(self, image, confidence_input):
        """
        Detects a face in the image and returns the region of interest (ROI).

        Parameters:
        -----------
        image : str or numpy.ndarray
            Path to the image file or an image array.
        confidence_input : float
            The minimum confidence threshold for detecting a face.

        Returns:
        --------
        tuple
            A tuple containing the ROI (numpy.ndarray) and the confidence score (float).
            Returns (None, None) if no face is detected with sufficient confidence.
        """
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
