import unittest
from main.processor import Processor
import cv2

class Boundary_testcase(unittest.TestCase):

    def setUp(self):
        self.img = cv2.imread('../test.png')
        self.proc = Processor()

    def test_b(self):
        self.assertTrue(self.img != None)
        self.proc.set_boundary(self.img)

    def tearDown(self):
        pass