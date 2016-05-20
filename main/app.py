import serial
import cv2
import numpy as np
from .commander import Commander
from .processor import Processor

def nothing(x):
    pass

class App(object):

    def __init__(self, port='/dev/cu.HC-05-DevB', baud=9600, cam=0):
        #self.commander = Commander(port, baud)
        print('Commander initialized.')

        self.camera    = cv2.VideoCapture(cam)
        print('Camera initialized')

        self.processer = Processor()

    def adjust_Cam(self):
        print('Adjust camera position. Enter q to continue.')
        while True:
            ret, frame = self.camera.read()

            if ret:
                cv2.imshow("Test", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        print('Camera position confirmed.')

    def configure(self):
        self.adjust_Cam()
        self.processer.set_boundary(self.camera)
        print('done!')
    def update(self):
        pass

    def run(self):
        self.configure()
        
        cv2.namedWindow('webcam')
        cv2.namedWindow('persp')
        cv2.namedWindow('thresh')
        cv2.namedWindow('thin')
        cv2.namedWindow('corner')
        cv2.namedWindow('line')

        # cv2.createTrackbar('cnt', 'corner', 1,25, nothing)

        cv2.createTrackbar('th', 'thresh', 0,255, nothing)

        cv2.createTrackbar('minLL', 'line', 0, 100, nothing)
        cv2.createTrackbar('maxLG', 'line', 0, 100, nothing)
        ret, frame = self.camera.read()

        while True:
            # ret, frame = self.camera.read()

            th = cv2.getTrackbarPos('th', 'thresh')
            points = cv2.getTrackbarPos('cnt', 'corner')
            minLL = cv2.getTrackbarPos('minLL', 'line')
            maxLG = cv2.getTrackbarPos('maxLG', 'line')

            # perspective tranform
            warp = self.processer.perspectiveTransform(frame)
            cv2.imshow('persp', warp)

            # thresh
            thresh = self.processer.thresh(warp, th)
            cv2.imshow('thresh', thresh)

            # thinning
            thinned = self.processer.thin(thresh)
            cv2.imshow('thin', thinned)
            #
            # corner = self.processer.corner(thinned, points, warp)
            # cv2.imshow('corner', corner)

            line = self.processer.find_lines(thinned, warp, minLL, maxLG)
            cv2.imshow('line', line)

            k = cv2.waitKey(20) & 0xFF


            if k == ord('q'):
                cv2.imwrite('photo.png', warp)
                break
            #self.commander.send_cmd(chr(k))
        cv2.destroyAllWindows()
        print('END.')
        #self.commander.disconnect()

