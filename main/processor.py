import cv2
import numpy as np
from thin import thinning
import math
from utils import *
import time
USE_FILE = False
# USE_FILE = True

USE_CAR = False
USE_CAR = True

MAXX = 400
MAXY = 200
delta = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

frame = None
done = False
roiPts_head = []
roiPts_tail = []
inputMode = None

def valid(x, y):
    return 0 <= x < MAXX and 0 <= y < MAXY

def nothing(x):
    pass

    frame = None
    roiPts_head = []
    roiPts_tail = []
    done = False
    inputMode = False

def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts_head, done, roiPts_tail, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and (len(roiPts_head) < 4 or len(roiPts_tail) < 4):
        if len(roiPts_head) < 4:
            roiPts_head.append((x, y))
        else:
            roiPts_tail.append((x, y))
            if len(roiPts_tail) == 4:
                done = True
        cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


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

    def set_car(self, img):
        frame = img.copy()
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
        (ret, frame) = camera.read()
        self.set_car(self.perspectiveTransform(frame))
        currentCorner = 1
        while True:
            (grabbed, frame) = camera.read()

            if not grabbed:
                break
            frame = self.perspectiveTransform(frame)
            blurred = cv2.GaussianBlur(frame, (11,11), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

                center = centerPt(head_center, tail_center)

                dir = unitVec(tail_center, head_center)

                ndir = unitVec(center, self.corners[currentCorner])

                distance = dist(center, self.corners[currentCorner])
                bias = cross(dir, ndir)

                print "center: ", center
                print "dir: ", dir
                print "ndir: ", ndir
                print "bias: ", bias
                print "dist: ", distance
                print '=' * 80, '\n'

                if abs(bias) < 0.5:
                    cmd.send_cmd('w')
                elif bias < 0:
                    cmd.send_cmd('a')
                elif bias > 0:
                    cmd.send_cmd('d')
                if distance < 40:
                    currentCorner += 1
                    print '*' * 10, ' %d ' % currentCorner, '*' * 10
                    if currentCorner == len(self.corners):
                        break

            cv2.imshow("mask", head_mask)
            cv2.imshow("frame", frame)
            time.sleep(1.5)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break


        cv2.destroyAllWindows()





    def main(self, camera, cmd):
        # grab the reference to the current frame, list of ROI
        # points and whether or not it is ROI selection mode
        global frame, done, roiPts_head, roiPts_tail, inputMode

        # setup the mouse callback
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", selectROI)

        # initialize the termination criteria for cam shift, indicating
        # a maximum of ten iterations or movement by a least one pixel
        # along with the bounding box of the ROI
        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        roiBox_head = None
        roiBox_tail = None


        head = None
        tail = None
        center = None
        dir = None

        currentCorner = 1
        # keep looping over the frames
        while True:
            # grab the current frame
            (grabbed, frame) = camera.read()
            frame = self.perspectiveTransform(frame)
            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break

            # if the see if the ROI has been computed
            if roiBox_head is not None:
                # convert the current frame to the HSV color space
                # and perform mean shift
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                backProj = cv2.calcBackProject([hsv], [0], roiHist_head, [0, 180], 1)

                # apply cam shift to the back projection, convert the
                # points to a bounding box, and then draw them
                (r, roiBox_head) = cv2.CamShift(backProj, roiBox_head, termination)
                pts = np.int0(cv2.boxPoints(r))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx = (pts[0][0] + pts[2][0]) / 2
                cy = (pts[0][1] + pts[2][1]) / 2
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), 2)

                head = (cx, cy)

            if roiBox_tail is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                backProj = cv2.calcBackProject([hsv], [0], roiHist_tail, [0, 180], 1)

                # apply cam shift to the back projection, convert the
                # points to a bounding box, and then draw them
                (r, roiBox_tail) = cv2.CamShift(backProj, roiBox_tail, termination)
                pts = np.int0(cv2.boxPoints(r))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx = (pts[0][0] + pts[2][0]) / 2
                cy = (pts[0][1] + pts[2][1]) / 2
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), 2)

                tail = (cx, cy)

            # show the frame and record if the user presses a key
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if roiBox_tail is not None and roiBox_head is not None:

                center = centerPt(head, tail)

                dir = unitVec(tail, head)


                ndir = unitVec(center, self.corners[currentCorner].ravel())

                distance = dist(center, self.corners[currentCorner].ravel())
                bias = cross(dir, ndir)

                print "center: ", center
                print "dir: ", dir
                print "ndir: ", ndir
                print "bias: ", bias
                print "dist: ", distance
                print '='*80, '\n'

                if abs(bias) < 0.7:
                    cmd.send_cmd('w')
                elif bias < 0:
                    cmd.send_cmd('a')
                elif bias > 0:
                    cmd.send_cmd('d')
                if distance < 20:
                    currentCorner += 1
                    if currentCorner == len(self.corners):
                        break
                time.sleep(1)


            # handle if the 'i' key is pressed, then go into ROI
            # selection mode
            if key == ord("i") and (len(roiPts_head) < 4 or len(roiPts_tail) < 4):
                # indicate that we are in input mode and clone the
                # frame
                inputMode = True
                orig = frame.copy()

                # keep looping until 4 reference ROI points have
                # been selected; press any key to exit ROI selction
                # mode once 4 points have been selected
                while not done:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(0)

                # determine the top-left and bottom-right points
                roiPts_head = np.array(roiPts_head)
                roiPts_tail = np.array(roiPts_tail)




                s = roiPts_head.sum(axis = 1)
                tl = roiPts_head[np.argmin(s)]
                br = roiPts_head[np.argmax(s)]

                # grab the ROI for the bounding box and convert it
                # to the HSV color space
                roi_head = orig[tl[1]:br[1], tl[0]:br[0]]
                roi_head = cv2.cvtColor(roi_head, cv2.COLOR_BGR2HSV)
                #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                # mask = cv2.inRange(roi_head, np.array((72., 29., 97.)), np.array((162., 140., 184.)))
                # compute a HSV histogram for the ROI and store the
                # bounding box
                roiHist_head = cv2.calcHist([roi_head], [0], None, [180], [0, 180])
                roiHist_head = cv2.normalize(roiHist_head, roiHist_head, 0, 255, cv2.NORM_MINMAX)
                roiBox_head = (tl[0], tl[1], br[0], br[1])

                s = roiPts_tail.sum(axis=1)
                tl = roiPts_tail[np.argmin(s)]
                br = roiPts_tail[np.argmax(s)]

                # grab the ROI for the bounding box and convert it
                # to the HSV color space
                roi_tail = orig[tl[1]:br[1], tl[0]:br[0]]
                roi_tail = cv2.cvtColor(roi_tail, cv2.COLOR_BGR2HSV)
                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

                # mask = cv2.inRange(roi_tail, np.array((0., 141., 192.)), np.array((67., 202., 255.)))
                # compute a HSV histogram for the ROI and store the
                # bounding box
                roiHist_tail = cv2.calcHist([roi_tail], [0], None, [16], [0, 180])
                roiHist_tail = cv2.normalize(roiHist_tail, roiHist_tail, 0, 255, cv2.NORM_MINMAX)
                roiBox_tail = (tl[0], tl[1], br[0], br[1])

            # if the 'q' key is pressed, stop the loop
            elif key == ord("q"):
                cmd.write('xx')
                break

        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()






# if __name__ == '__main__':
#
#     p = Processor()
#     cv2.namedWindow('line')
#
#     # cv2.createTrackbar('cnt', 'corner', 1,25, nothing)
#
#     cv2.createTrackbar('minLL', 'line', 0, 100, nothing)
#     cv2.createTrackbar('maxLG', 'line', 0, 100, nothing)
#
#
#     test = cv2.imread('../test.png')
#
#     # thinned = p.thin(test)
#
#     bw = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#     _, bw2 = cv2.threshold(bw, 50, 255, cv2.THRESH_BINARY_INV)
#     thinned = thinning(bw2)
#     while True:
#         minLL = cv2.getTrackbarPos('minLL', 'line')
#         maxLG = cv2.getTrackbarPos('maxLG', 'line')
#         th1 = cv2.getTrackbarPos('th1', 'line')
#         th2 = cv2.getTrackbarPos('th2', 'line')
#
#         line = p.find_lines(thinned, thinned, minLL, maxLG, th1, th2)
#         cv2.imshow('line', line)
#         cv2.imshow('thin', thinned)
#
#         k = cv2.waitKey(20) & 0xFF
#
#         if k == ord('q'):
#             print minLL, maxLG
#             break
#             # self.commander.send_cmd(chr(k))
#     cv2.destroyAllWindows()
    