import cv2
import numpy as np
from .thin import thinning
import math

USE_FILE = False
USE_FILE = True
class Processor(object):

    def __init__(self):
        self.shape = np.float32([[0,0],[800,0],[800,400],[0,400]])
        self.points = np.float32([[0,0],[400,0],[400,200],[0,200]])
        self.points_cnt = 0

        self.perspectiveMatrix = None

    def set_boundary(self, cam):
        if USE_FILE:
            ff = open("boundary.txt", 'r')
            for i in range(4):
                for j in range(2):
                    n = ff.readline()
                    n = float(n)
                    self.points[i][j] = n
        else:
            active = False
            def get(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and active:
                    self.points[self.points_cnt, 0] = x
                    self.points[self.points_cnt, 1] = y
                    self.points_cnt += 1
                    print(x,y)
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
            ff = open("boundary.txt", 'w')
            for row in self.points:
                for j in row:
                    ff.write(str(j))
                    ff.write('\n')

        self.perspectiveMatrix = cv2.getPerspectiveTransform(self.points, self.shape)

    def thresh(self, src, th):
        ret, thresh1 = cv2.threshold(src, th, 255, cv2.THRESH_BINARY_INV)
        return thresh1

    def perspectiveTransform(self, img):
        return cv2.warpPerspective(img, self.perspectiveMatrix, (800,400))

    def corner(self, src, points, toshow):
        show = toshow.copy()
        img = src.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img, points, 0.01,10)

        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(show, (x,y), 3, 255, -1)

        return show

    def thin(self, src):
        bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return thinning(bw)

    def find_lines(self, src, toshow, minLL, maxLG):
        show = toshow.copy()
        ret, img = cv2.threshold(src, 20, 255, cv2.THRESH_BINARY_INV)

        dst = cv2.Canny(img, 50, 200)
        cv2.imshow('canny', dst)
        lines = cv2.HoughLines(dst, 1, math.pi/180.0, 60)

        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0,y0 = a*rho, b*rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(show, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

        # gray = img.copy()
        # edges = cv2.Canny(gray, 50, 200, apertureSize=3)
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLL, maxLG)
        # if lines is not None:
        #     for x1, y1, x2, y2 in lines[0]:
        #         cv2.line(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return show

if __name__ == '__main__':
    img = cv2.imread('../test.png')
    p = Processor()
    p.thresh(img)