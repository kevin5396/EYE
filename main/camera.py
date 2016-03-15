import cv2
import numpy

class Camera(object):

    def __init__(self, cam):
        self.cap = cv2.VideoCapture(cam)

    def read(self):
        ret, frame = self.cap.read()
