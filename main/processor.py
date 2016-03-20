import cv2
import numpy as np

class Processor(object):

    def __init__(self, shape=(300,300)):
        self.shape = shape
        self.points = np.float32([[0,0],[0,0],[0,0],[0,0]])
        self.points_cnt = 0

    def set_boundary(self, cam):
        active = False
        def get(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and active:
                self.points[self.points_cnt, 0] = x
                self.points[self.points_cnt, 1] = y
                self.points_cnt += 1
                print x,y
        cv2.namedWindow('getBoundary')
        cv2.setMouseCallback('getBoundary', get)
        while True:
            ret, img = cam.read()
            if ret:
                cv2.imshow('getBoundary', img)
            k = cv2.waitKey(20) & 0xFF
            if k == 32:
                active = not active
            if k == 27:
                break
            if self.points_cnt == 4:
                break
        cv2.destroyWindow('getBoundary')
        for row in self.points:
            print(row)

    def thresh(self, src):
        ret, thresh1 = cv2.threshold(src, 50, 255, cv2.THRESH_BINARY_INV)
        cv2.namedWindow("thresh")
        while True:
            cv2.imshow("thresh", thresh1)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        return thresh1

if __name__ == '__main__':
    img = cv2.imread('../test.png')
    p = Processor()
    p.thresh(img)