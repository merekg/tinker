from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():

    # check the number of inputs
    if(len(sys.argv) != 2):
        print("USE: python LoadDeadPixMap.py <input_path.tiff>")

    inPath = sys.argv[1]
    outPath = "deadPixMap.txt"
    
    # open the image
    im  = Image.open(inPath)

    # turn it into a numpy array
    imarray = np.array(im)

    # Zip the bad pixels into ordered pairs
    badPixels = np.where(imarray)
    badPixels = zip(badPixels[1], badPixels[0])

    # print some info for the user
    print("Total pixels in panel: " + str(len(imarray)*len(imarray[0])))
    print("Bad pixel count: " + str(len(badPixels)))

    # Save
    print("Saving to " + str(outPath)+ "...")
    outfile = open(outPath, "w")
    for pixel in badPixels:
        line = "[" + str(pixel[0]) + "," + str(pixel[1]) + "]\n"
        outfile.write(line)

    print("Done.")

if(__name__ == "__main__"):
    main()
