#!/usr/bin/env python -B
__author__ = 'Arvi Cheryauka'
import sys,getopt

##############################################################################
#
# Handle command line I/O

def main(argv):
    phanfile = ''
    imagefile = ''
    calibfile = ''
    recovfile = ''
    try:
      opts, args = getopt.getopt(argv,"hp:i:c:r:",["pfile=","ifile=","cfile=","rfile="])
    except getopt.GetoptError:
      print 'Use: python script.py -p <phanfile> -i <imagefile> -c <calibfile> -r <recovfile>'
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'Use: python script.py -p <phanfile> -i <imagefile> -c <calibfile> -r <recovfile>'
            sys.exit()
        elif opt in ("-p", "--pfile"):
            phanfile = arg
        elif opt in ("-i", "--ifile"):
            imagefile = arg
        elif opt in ("-c", "--cfile"):
            calibfile = arg
        elif opt in ("-r", "--rfile"):
            recovfile = arg
    return phanfile,imagefile,calibfile,recovfile

