import math

TRIM = 20
def dist2(x0, y0, x1, y1):
    return (x0-x1)**2 + (y0-y1)**2


def dist(p1, p2):
    return math.sqrt(dist2(p1[0],p1[1], p2[0],p2[1]))


def centerPt(p1, p2):
    return (int(p1[0]+p2[0])/2, int(p1[1]+p2[1])/2)

def unitVec(p1, p2):
    d = dist(p1, p2)
    vec = ((p2[0]-p1[0])/d, (p2[1]-p1[1])/d)
    return vec

def cross(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def trim(p1, p2):
    return ((p1[0],p1[1]+TRIM),(p2[0],p2[1]+TRIM))