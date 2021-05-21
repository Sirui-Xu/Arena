import sys
import os
 
class Logger(object):
    def __init__(self, filename="log.txt", mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.log.flush()