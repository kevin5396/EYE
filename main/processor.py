import cv2
import numpy as np
from thin import thinning
import math
from utils import *
import time
USE_FILE = False
USE_FILE = True

USE_CAR = False
# USE_CAR = True
#
DELAY_TYPE = 0
MAXX = 400
MAXY = 200
delta = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]


def valid(x, y):
    return 0 <= x < MAXX and 0 <= y < MAXY

def nothing(x):
    pass



def filter_points(corners, thresh):
    if len(corners) < 3:
        return corners
    filtered = corners[:2]
    for i in range(2, len(corners)):
        p1 = filtered[-2]
        p2 = filtered[-1]
        p3 = corners[i]
        res = (p2[1]-p1[1])*(p3[0]-p2[0]) - (p2[0]-p1[0])*(p3[1]-p2[1])
        if abs(res) <= thresh:
            del filtered[-1]
        filtered.append(p3)
    return filtered


class Processor(object):

    def __init__(self):
        self.shape = np.float32([[0,0],[MAXX,0],[MAXX,MAXY],[0,MAXY]])
        self.points = np.float32([[0,0],[MAXX,0],[MAXX,MAXY],[0,MAXY]])
        self.points_cnt = 0

        self.perspectiveMatrix = None

        self.dist = np.zeros(shape=(MAXX,MAXY))
        self.visited = np.zeros(shape=(MAXX,MAXY))
        self.hsv = [[0,0,0,0,0,0],[0,0,0,0,0,0]]

        self.head = (0,0)
        self.tail = (0,0)

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
        return cv2.warpPerspective(img, self.perspectiveMatrix, (MAXX,MAXY))

    def corner(self, src, points, toshow):
        show = toshow.copy()
        img = src.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img, points, 0.01,10)

        corners = np.int0(corners)
        self.corners = corners.copy()
        self.graph = toshow.copy()
        for i in corners:
            x, y = i.ravel()
            cv2.circle(show, (x,y), 3, 255, -1)

        return show
    def process_corners(self):
        self.startx = 0
        self.starty = 0
        def getPoint(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.startx = x
                self.starty = y
                print x,y

        cv2.namedWindow('getStart')
        cv2.setMouseCallback('getStart', getPoint)
        while True:
            cv2.imshow('getStart', self.graph)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()

        best = 10000000
        bestIndex = 0
        # find the closest point to the mouse picked point
        for i in range(len(self.corners)):
            d = dist2(self.startx, self.starty, self.corners[i].ravel()[0], self.corners[i].ravel()[1])
            if d < best:
                bestIndex = i
                best = d

        self.startx, self.starty = self.corners[bestIndex].ravel()
        print self.startx, self.starty
        # print self.graph[self.starty, self.startx]

        self.cal_dist()


    def cal_dist(self):
        queue = [(self.startx, self.starty, 0)]
        while queue:
            x, y, d = queue.pop(0)
            if self.visited[x,y] == 1:
                continue
            self.visited[x,y] = 1
            self.dist[x,y] = d
            for (dx,dy) in delta:
                newx = x + dx
                newy = y + dy
                if valid(newx, newy) and self.graph[newy, newx] == 255 and self.visited[newx, newy] == 0:
                    queue.append((newx, newy, d+1))
        #
        # for i in self.corners:
        #     x,y = i.ravel()
        #     print x,y, self.dist[x,y]

        gg = cv2.cvtColor(self.graph, cv2.COLOR_GRAY2BGR)
        kk = gg.copy()

        for i in self.corners:
            x, y = i.ravel()
            cv2.circle(gg, (x,y), 2+int(self.dist[x,y]/50),(0,0,255), -1)

        corners_list = [i.ravel().tolist() for i in self.corners]
        corners_list.sort(key=lambda li: self.dist[li[0], li[1]])
        corners_list = filter_points(corners_list, 300)
        for i in range(len(corners_list)):
            cv2.circle(kk, (corners_list[i][0],corners_list[i][1]), 2+int(i/2),(0,0,255), -1)

        self.corners = corners_list


        cv2.imshow("gg", gg)
        cv2.imshow("kk", kk)
        while True:
            k = cv2.waitKey(20) & 0xFF

            if k == ord('q'):
                break

        cv2.destroyAllWindows()










    def thin(self, src):
        # bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return thinning(src)


    def set_car(self, camera):

        if USE_CAR:
            ff = open("car.txt", 'r')
            for j in range(2):
                for i in range(6):
                    self.hsv[j][i] = int(ff.readline())
        else:

            cv2.namedWindow("frame")
            cv2.createTrackbar('hmin', 'frame', 0, 255, nothing)
            cv2.createTrackbar('hmax', 'frame', 0, 255, nothing)
            cv2.createTrackbar('smin', 'frame', 0, 255, nothing)
            cv2.createTrackbar('smax', 'frame', 0, 255, nothing)
            cv2.createTrackbar('vmin', 'frame', 0, 255, nothing)
            cv2.createTrackbar('vmax', 'frame', 0, 255, nothing)
            cnt = 0
            while cnt < 2:
                ret, frame = camera.read()
                frame = self.perspectiveTransform(frame)

                self.hsv[cnt][0] = cv2.getTrackbarPos('hmin', 'frame')
                self.hsv[cnt][1] = cv2.getTrackbarPos('hmax', 'frame')
                self.hsv[cnt][2] = cv2.getTrackbarPos('smin', 'frame')
                self.hsv[cnt][3] = cv2.getTrackbarPos('smax', 'frame')
                self.hsv[cnt][4] = cv2.getTrackbarPos('vmin', 'frame')
                self.hsv[cnt][5] = cv2.getTrackbarPos('vmax', 'frame')

                # Convert BGR to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # define range of blue color in HSV
                lower_blue = np.array([self.hsv[cnt][0], self.hsv[cnt][2], self.hsv[cnt][4]])
                upper_blue = np.array([self.hsv[cnt][1], self.hsv[cnt][3], self.hsv[cnt][5]])
                # lower_blue = np.array([200,48,40])
                # upper_blue = np.array([220,55,50])

                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                cv2.imshow('frame', mask)
                cv2.imshow('origin', hsv)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
                if k == ord('n'):
                    cnt += 1
            
            ff = open("car.txt", "w")
            for cnt in range(2):
                for i in range(6):
                    ff.write(str(self.hsv[cnt][i]))
                    ff.write('\n')
            cv2.destroyAllWindows()
            
        self.lower_head = np.array([self.hsv[0][0], self.hsv[0][2], self.hsv[0][4]])
        self.upper_head = np.array([self.hsv[0][1], self.hsv[0][3], self.hsv[0][5]])
        
        
        self.lower_tail = np.array([self.hsv[1][0], self.hsv[1][2], self.hsv[1][4]])
        self.upper_tail = np.array([self.hsv[1][1], self.hsv[1][3], self.hsv[1][5]])
        print self.hsv

    def work(self, camera, cmd):
        self.set_car(camera)
        currentCorner = 1
        while True:
            DELAY_TYPE = 0
            cmd.send_cmd('x')
            (grabbed, frame) = camera.read()


            if not grabbed:
                break
            frame = self.perspectiveTransform(frame)
            blurred = cv2.GaussianBlur(frame, (11,11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # find head
            head_mask = cv2.inRange(hsv, self.lower_head, self.upper_head)
            head_mask = cv2.erode(head_mask, None, iterations=2)
            head_mask = cv2.dilate(head_mask, None, iterations=2)
            head_cnts = cv2.findContours(head_mask.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]

            head_center = None
            if len(head_cnts) > 0:
                c = max(head_cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                head_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 3:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255,255),2)
                    cv2.circle(frame, head_center, 5, (0,0,255), -1)

            tail_mask = cv2.inRange(hsv, self.lower_tail, self.upper_tail)
            tail_mask = cv2.erode(tail_mask, None, iterations=2)
            tail_mask = cv2.dilate(tail_mask, None, iterations=2)
            tail_cnts = cv2.findContours(tail_mask.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]

            tail_center = None
            if len(tail_cnts) > 0:
                c = max(tail_cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                tail_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 3:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(frame, tail_center, 5, (0, 0, 255), -1)

            cc = self.corners[currentCorner]
            cv2.circle(frame, (cc[0], cc[1]), 5, (255,255,0), -1)
            if head_center is not None and tail_center is not None:
                DELAY_TYPE = 0
                head_center, tail_center = trim(head_center, tail_center)
                center = centerPt(head_center, tail_center)
                try:
                    dir = unitVec(tail_center, head_center)

                    ndir = unitVec(center, self.corners[currentCorner])

                    distance = dist(center, self.corners[currentCorner])
                    bias = cross(dir, ndir)
                except ZeroDivisionError, e:
                    pass
                else:
                    print "center: ", center
                    print "dir: ", dir
                    print "ndir: ", ndir
                    print "bias: ", bias
                    print "dist: ", distance
                    print '=' * 80, '\n'
                    if distance < 15:
                        currentCorner += 1
                        DELAY_TYPE = -1
                        print '*' * 10, ' %d ' % currentCorner, '*' * 10
                        if currentCorner == len(self.corners):
                            break
                    elif bias > -0.2 and bias < 0.1:
                        if distance >= 15:
                            print 'w'
                            cmd.send_cmd('w')
                            DELAY_TYPE = 1
                        else:
                            currentCorner += 1
                            print '*' * 10, ' %d ' % currentCorner, '*' * 10
                            if currentCorner == len(self.corners):
                                break
                    elif bias < 0:
                        print 'a'
                        cmd.send_cmd('a')
                    elif bias > 0:
                        print 'd'
                        cmd.send_cmd('d')

            else:
                cmd.send_cmd('x')

            cv2.imshow("mask", head_mask)
            cv2.imshow("mask1", tail_mask)
            cv2.imshow("frame", frame)
            if DELAY_TYPE == 0:
                time.sleep(0.4)
            elif DELAY_TYPE == 1:
                time.sleep(0.6)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break

        cmd.send_cmd('x')
        cv2.destroyAllWindows()
