import serial
from commander import Commander
from camera import Camera
class App(object):

    def __init__(self, port='/dev/cu.HC-05-DevB', baud=9600, cam=0):
        self.commander = Commander(port, baud)
        self.camera    = Camera(cam)

    def configure(self):
        pass

    def update(self):
        pass

    def run(self):
        self.configure()
        while True:
            self.update()

