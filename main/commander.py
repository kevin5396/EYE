import serial
class Commander(object):

    def __init__(self, port, baudrate):
        self.conn = serial.Serial(port, baudrate)
        if not self.conn.is_open:
            raise Exception("Unable to open port %s at baudrate: %s" % (port, baudrate))

    def make_cmd(self, position, course_map):
        pass

    def send_cmd(self, cmd):
        try:
            self.conn.write(cmd)
        except:
            print("Unable to send command.")