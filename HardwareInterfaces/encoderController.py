import serial
import time
import os
import sys

SERIAL_PORT_STRING = "/dev/serial/by-id/usb-RLS_Merilna_tehnika_E201_SSI_Encoder_Interface_48DD7F8A3133-if00"
GET_POSE_COMMAND = ">"
GET_STATUS_COMMAND = "p"
BAUD_RATE = 9600
ERROR_STRING = "ERROR"
OFFSET = 1073873565
BITS_TO_MM = 200.0/790.0

def sendCommand(cmd):
    pass

def readline():
    pass

def reportLoop(encoder):
    while True:
        try:
            encoder.flushInput()
            encoder.flushOutput()
            encoder.write(">\r\n")
            ans = encoder.readline()
            print(ans.replace("\r",'').replace("\n",''))
            pose = int(ans.replace(ERROR_STRING, ''),16) - OFFSET
            print(pose)
            if pose > 855 or pose < -50:
                continue
            print(pose*BITS_TO_MM)
        except:
            print("error in reportLoop. closing...")
            exit()

def continuous(encoder):
    encoder.flushInput()
    encoder.flushOutput()
    encoder.write("1\r\n")
    for i in range(100):
        print(encoder.readline().rstrip())

    encoder.write("0\r\n")
def main():

    # Set up the serial port
    encoder = serial.Serial(SERIAL_PORT_STRING, BAUD_RATE)
    encoder.bytesize = serial.EIGHTBITS
    encoder.parity = serial.PARITY_NONE
    encoder.stopbits = serial.STOPBITS_ONE
    encoder.timeout = 1
    encoder.xonxoff = False
    encoder.rtscts = False
    encoder.dsrdtr = False
    encoder.writeTimeout = 0

    #try:
        #encoder.open()
    #except Exception, e:
        #print("Error opening port: " +str(e))
        #exit()

    if not encoder.isOpen():
        print("Cannot open serial port.")
        print("Exiting...")
        exit()

    quit = False
    while not quit:
        cmd = raw_input(">")
        if cmd == 'q':
            quit=True
            continue
        elif cmd == 'l':
            reportLoop(encoder)
        elif cmd == '1':
            continuous(encoder)
        try:
            encoder.flushInput()
            encoder.flushOutput()

            encoder.write(cmd)
            print(encoder.readline().replace('\r', '\n'))
        except:
            print("error")
            exit()

if __name__ == "__main__":
    main()
