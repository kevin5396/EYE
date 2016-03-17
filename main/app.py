import serial
import cv2
import numpy as np
from commander import Commander
from processor import Processor
class App(object):

    def __init__(self, port='/dev/cu.HC-05-DevB', baud=9600, cam=0):
        #self.commander = Commander(port, baud)
        #print 'Commander initialized.'

        self.camera    = cv2.VideoCapture(cam)
        print 'Camera initialized'

        self.processer = Processor()

    def adjust_Cam(self):
        print 'Adjust camera position. Enter q to continue.'
        while True:
            ret, frame = self.camera.read()

            if ret:
                cv2.imshow("Test", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        print 'Camera position confirmed.'

    def configure(self):
        self.adjust_Cam()
        self.processer.set_boundary(self.camera)

    def update(self):
        pass

    def run(self):
        self.configure()
        while True:
            self.update()

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        print 'END.'

