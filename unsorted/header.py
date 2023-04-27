#!/usr/bin/env python -B
__author__ = 'Arvi Cheryauka'
import nrrd

# Handle NRRD / NHDR header i/on maps, in AcqHome + GainMap directory

def read( nrrdfile ):
    with open(nrrdfile,'rb') as filehandle:
        header = nrrd.read_header(filehandle)
    encoding= header["encoding"] 
    sizes= header["sizes"]
    datafile= header["data file"]
    endian= header["endian"]  
    dtype= header["type"]  
    dimension= header["dimension"]
    return encoding,sizes,datafile,endian,dtype,dimension

def write( filename,encoding,sizes,datafile,endian,dtype,dimension ):
    filehandle = open(filename, 'wb')
    filehandle.write('NRRD0004\n')
    filehandle.write('type: ' + dtype + '\n')
    filehandle.write('dimension: ' + dimension + '\n')
    filehandle.write('sizes: ' + sizes + '\n')
    filehandle.write('endian: ' + endian + '\n')
    filehandle.write('encoding: ' + encoding + '\n')
    filehandle.write('data file: ' + datafile + '\n')
    

