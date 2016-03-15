import numpy as np
import cv2
import unittest

class Webcam_testcase(unittest.TestCase):

    def setUp(self):
        self.cap = cv2.VideoCapture(0)

    def test_webcam(self):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()


            # Display the resulting frame
            if ret:
                cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def tearDown(self):
        self.cap.release()
        cv2.destroyAllWindows()