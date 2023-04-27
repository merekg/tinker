#! /usr/bin/env python3
import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import argparse

# Constants
GAIN_Z_SCORE_MIN = -10.0
GAIN_Z_SCORE_MAX = 10.0

LINEARITY_Z_SCORE_MIN = -10.5
LINEARITY_Z_SCORE_MAX = 15.0

# Take a single stack and find the low and high pixels. Return a tuple containing their locations.
def get_high_range_pixels(img_stack):
    low_img = np.min(img_stack, axis=0)
    high_img = np.max(img_stack, axis=0)
    range_img = high_img - low_img
    display_image(range_img)

def get_bad_gain_pixels(img_stack):

    avg_img = np.max(img_stack, axis=0)
    normal_img = (avg_img - avg_img.mean()) / avg_img.std()

    low_pixel_locations = np.where(normal_img <  GAIN_Z_SCORE_MIN)
    high_pixel_locations = np.where(normal_img > GAIN_Z_SCORE_MAX)

    print("Low gain pixel count:", low_pixel_locations[1].shape[0])
    print("High gain pixel count:", high_pixel_locations[1].shape[0])

    return list(zip(low_pixel_locations[1], low_pixel_locations[0])), list(zip(high_pixel_locations[1], high_pixel_locations[0]))
        
# Takes two stacks and finds pixels that are non-linear in gain. Return their locations.
def get_non_linear_pixels(medium_dose_img, high_dose_img):
    
    diff_img = np.max(high_dose_img, axis=0) - np.max(medium_dose_img, axis=0)
    normal_img = (diff_img - diff_img.mean()) / diff_img.std()
    
    non_linear_mask = np.logical_or(normal_img > LINEARITY_Z_SCORE_MAX, normal_img < LINEARITY_Z_SCORE_MIN)
    non_linear_pixel_locations = np.where(non_linear_mask)

    print("Non-linear pixel count:", len(non_linear_pixel_locations[0])) 

    return list(zip(non_linear_pixel_locations[1], non_linear_pixel_locations[0]))

def generate_dead_pixel_image(dead_pix_list, map_x, map_y):
    dead_pix_map = np.zeros((map_x,map_y))
    for coord in dead_pix_list:
        x_i = coord[0]
        y_i = coord[1]
        dead_pix_map[x_i,y_i] = 1
    return dead_pix_map

def display_image(img):
    plt.imshow(img)
    plt.show()

def parse_inputs(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output", help="File path of the output", required=True)
    parser.add_argument("-m","--mediumDoseImage", help="Path to the medium dose acquisition file", required=True)
    parser.add_argument('-hh','--highDoseImage', help="Path to the high dose acquisition file", required=True)

    parsed_list = parser.parse_args(args).__dict__

    # change relative paths to absolute paths
    medium_dose_path = os.path.abspath(parsed_list["mediumDoseImage"])
    high_dose_path = os.path.abspath(parsed_list["highDoseImage"])
    output_path = os.path.abspath(parsed_list["output"])
    if not os.path.exists(medium_dose_path):
        print("Error:",medium_dose_path,"does not exist. Exiting.")
        exit()
    if not os.path.exists(high_dose_path):
        print("Error:",high_dose_path,"does not exist. Exiting.")
        exit()

    return medium_dose_path, high_dose_path, output_path

def main():
    
    medium_dose_path, high_dose_path, output_path = parse_inputs(sys.argv[1:])

    with h5py.File(medium_dose_path, 'r') as stack_60_file, h5py.File(high_dose_path,'r') as stack_75_file:
        medium_dose_stack = stack_60_file['ITKImage/0/VoxelData'][:]
        high_dose_stack = stack_75_file['ITKImage/0/VoxelData'][:]
    
    # Check that the sizes are the same.
    assert medium_dose_stack.shape == high_dose_stack.shape
    size = medium_dose_stack.shape[1]*medium_dose_stack.shape[2]

    # Call the helper functions and store the data 
    get_high_range_pixels(high_dose_stack)
    low_gain_pixels,high_gain_pixels = get_bad_gain_pixels(high_dose_stack)
    non_linear_pixels = get_non_linear_pixels(medium_dose_stack,high_dose_stack)

    dead_pixels = set(low_gain_pixels + high_gain_pixels + non_linear_pixels)

    # Write the dead pixels to deadPixMap.txt in the working directory.
    output_file = open(output_path,"w")
    for pixel in dead_pixels:
        line = "[" + str(pixel[0]) + "," + str(pixel[1]) + "]"
        output_file.write(line + '\n')
        
    # Display the visualization of the dead pixels
    img = generate_dead_pixel_image(dead_pixels, 976,976)
    display_image(img.T)

if __name__ == '__main__':
    main()

