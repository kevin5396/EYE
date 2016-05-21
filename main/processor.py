import cv2
import numpy as np
from thin import thinning
import math
from utils import *
USE_FILE = False
USE_FILE = True

MAXX = 400
MAXY = 200
delta = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
def valid(x, y):
    return 0 <= x < MAXX and 0 <= y < MAXY

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
        self.shape = np.float32([[0,0],[400,0],[400,200],[0,200]])
        self.points = np.float32([[0,0],[400,0],[400,200],[0,200]])
        self.points_cnt = 0

        self.perspectiveMatrix = None

        self.dist = np.zeros(shape=(400,200))
        self.visited = np.zeros(shape=(400,200))

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
        return cv2.warpPerspective(img, self.perspectiveMatrix, (400,200))

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

        for i in self.corners:
            x,y = i.ravel()
            print x,y, self.dist[x,y]

        gg = cv2.cvtColor(self.graph, cv2.COLOR_GRAY2BGR)
        kk = gg.copy()

        for i in self.corners:
            x, y = i.ravel()
            cv2.circle(gg, (x,y), 2+int(self.dist[x,y]/50),(0,0,255), -1)

        corners_list = [i.ravel().tolist() for i in self.corners]
        corners_list.sort(key=lambda li: self.dist[li[0], li[1]])
        corners_list = filter_points(corners_list, 300)
        for i in range(len(corners_list)):
            cv2.circle(kk, (corners_list[i][0],corners_list[i][1]), 2+int(i/3),(0,0,255), -1)




        cv2.imshow("gg", gg)
        cv2.imshow("kk", kk)
        while True:
            k = cv2.waitKey(20) & 0xFF

            if k == ord('q'):
                # cv2.imwrite('photo.png', warp)
                break

        cv2.destroyAllWindows()










    def thin(self, src):
        # bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return thinning(src)

    def find_lines(self, src, toshow):
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
        #
        # gray = src.copy()
        # edges = cv2.Canny(gray, th1, th2, apertureSize=5)
        # cv2.imshow("edge", edges)
        #
        # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=1, minLineLength=minLL, maxLineGap=maxLG)
        # if lines is not None:
        #     for x1, y1, x2, y2 in lines[0]:
        #         cv2.line(show, (x1, y1), (x2, y2), (0, 0, 255), 3, 8)
        return show

    def setup(self):
        pass



def nothing(x):
    pass

if __name__ == '__main__':

    p = Processor()
    cv2.namedWindow('line')

    # cv2.createTrackbar('cnt', 'corner', 1,25, nothing)

    cv2.createTrackbar('minLL', 'line', 0, 100, nothing)
    cv2.createTrackbar('maxLG', 'line', 0, 100, nothing)


    test = cv2.imread('../test.png')

    # thinned = p.thin(test)

    bw = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    _, bw2 = cv2.threshold(bw, 50, 255, cv2.THRESH_BINARY_INV)
    thinned = thinning(bw2)
    while True:
        minLL = cv2.getTrackbarPos('minLL', 'line')
        maxLG = cv2.getTrackbarPos('maxLG', 'line')
        th1 = cv2.getTrackbarPos('th1', 'line')
        th2 = cv2.getTrackbarPos('th2', 'line')

        line = p.find_lines(thinned, thinned, minLL, maxLG, th1, th2)
        cv2.imshow('line', line)
        cv2.imshow('thin', thinned)

        k = cv2.waitKey(20) & 0xFF

        if k == ord('q'):
            print minLL, maxLG
            break
            # self.commander.send_cmd(chr(k))
    cv2.destroyAllWindows()