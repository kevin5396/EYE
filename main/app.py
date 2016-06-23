import serial
import cv2
import numpy as np
from .commander import Commander
from .processor import Processor

def nothing(x):
    pass

class App(object):

    def __init__(self, port='/dev/cu.HC-05-DevB', baud=9600, cam=0):
        self.commander = Commander(port, baud)
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
        try:
            self.configure()

            cv2.namedWindow('webcam')
            cv2.namedWindow('persp')
            cv2.namedWindow('thresh')
            cv2.namedWindow('thin')
            cv2.namedWindow('corner')

            cv2.createTrackbar('th', 'thresh', 120,255, nothing)

            ret, frame = self.camera.read()
            while True:

                th = cv2.getTrackbarPos('th', 'thresh')

                # perspective tranform
                warp = self.processer.perspectiveTransform(frame)
                warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

                warp = cv2.GaussianBlur(warp, (5,5), 0)
                cv2.imshow('persp', warp)
                # thresh
                thresh = self.processer.thresh(warp, th)
                cv2.imshow('thresh', thresh)

                #thinning
                thinned = self.processer.thin(thresh)
                cv2.imshow('thin', thinned)

                # # #
                corner = self.processer.corner(thinned, 25, thresh)
                cv2.imshow('corner', corner)


                k = cv2.waitKey(20) & 0xFF


                if k == ord('q'):
                    break

            cv2.destroyAllWindows()
            self.processer.process_corners()
            self.processer.work(self.camera, self.commander)
            self.camera.release()
            print('END.')
        except Exception, e:
            print e
        finally:
            self.commander.disconnect()


