"""
Resources:
https://www.askpython.com/python/examples/display-images-using-python
https://realpython.com/image-processing-with-the-python-pillow-library/
https://www.martinreddy.net/gfx/2d-hi.html
https://www.researchgate.net/figure/8-bit-256-x-256-Grayscale-Lena-Image_fig1_3935609
https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
https://www.geeksforgeeks.org/python-opencv-getting-and-setting-pixels/
https://www.codesansar.com/numerical-methods/linear-interpolation-python.htm

Gabriel Alfredo Siguenza, CS 5550 Digital Image Processing
Hw 2
Dr. Amar Raheja
Date: 10-09-2023
"""

# Importing Libraries
import cv2
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

global original_image, processed_image_label


def linear_interpolation(image, zoom_factor_height):
    height, width = image.shape
    new_height = int(height * zoom_factor_height)

    # Initialize the zoomed image
    zoomed_image = np.zeros((new_height, width), dtype=np.uint8)

    for i in range(new_height):
        # Compute the corresponding row in the original image
        old_i = i / zoom_factor_height

        # Nearest neighboring rows
        y1, y2 = int(old_i), min(int(old_i) + 1, height - 1)

        # Linear interpolation along the height dimension
        dy = old_i - y1
        interpolated_row = (1 - dy) * image[y1, :] + dy * image[y2, :]

        zoomed_image[i, :] = interpolated_row

    return zoomed_image


def bilinear_interpolation(image, zoom_factor_height, zoom_factor_width):
    height, width = image.shape
    new_height = int(height * zoom_factor_height)
    new_width = int(width * zoom_factor_width)

    # Initialize the zoomed image
    zoomed_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # Compute the coordinates in the original image
            old_i = i / zoom_factor_height
            old_j = j / zoom_factor_width

            # Nearest neighboring pixel coordinates
            x1, y1 = int(old_j), int(old_i)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

            # Interpolate using bilinear interpolation
            dx = old_j - x1
            dy = old_i - y1
            interpolated_value = (1 - dx) * (1 - dy) * image[y1, x1] + \
                                 dx * (1 - dy) * image[y1, x2] + \
                                 (1 - dx) * dy * image[y2, x1] + \
                                 dx * dy * image[y2, x2]

            zoomed_image[i, j] = int(interpolated_value)

    return zoomed_image


def nearest_neighbor_interpolation(image, zoom_factor_height, zoom_factor_width):
    width = int(image.shape[1])
    height = int(image.shape[0])
    new_height = int(height * zoom_factor_height)
    new_width = int(width * zoom_factor_width)

    # init zoomed img
    zoomed_image = np.zeros((new_height, new_width), dtype=np.uint8)
    # zoomed_image = np.zeros(new_height, new_width)
    # zoomed_image = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            old_i = int(i / zoom_factor_height)
            old_j = int(j / zoom_factor_width)
            zoomed_image[i, j] = image[old_i, old_j]

        # print("zoomed_image shape:", zoomed_image.shape)
        # print("zoomed_image type:", type(zoomed_image))

    return zoomed_image


'''
1.
Vary the spatial resolution of this image from the given scale to 512x512 and down to 32x32 and then zoom it again to 
see the loss of detail. 

Use the nearest neighbor method, linear method (both x or y) and bi-linear interpolation method 
for zooming it back the desired resolution.
'''

'''
2.	
Vary the gray level resolution of your image from 8-bit to a 1-bit image in steps of 1-bits. Let the user decide 
how many number of bits or provide a selection from a drop-down menu.
'''

# In case of local histogram equalization, ask the user for the resolution of the square mask to be used.

def histogram_equalization():

    return

# Function to perform bit plane slicing for image
def reduce_gray_resolution(image, bits):
    img = np.asarray(image)

    # Extract the specified bit plane
    # max_pixel_value = (image >> bits) & 1
    L = 2 ** bits
    # mathematical relation between gray level resolution and bits per pixel
    # L = 2^k
    q = 256 / L
    q_image = (img / L).astype(np.uint8)
    q_image = ((q_image / (L - 1)) * 255).astype(np.uint8)

    # Scale the pixel values to the specified number of bits
    # reduced_resolution_image = (max_pixel_value * (image / 255)).astype(np.uint8)

    # Calculate the maximum pixel value for the specified number of bits
    # max_pixel_value = 2 ** bit_plane - 1

    # Multiply by 255 to visualize the bit plane
    # reduced_resolution_image = bit_plane_image * 255

    # return Image.fromarray(quantized_img)
    return q_image


# Function to open an image and process it using bit plane slicing
def process_bit_plane_slicing():
    global original_image, processed_image_label

    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    # Load the image using OpenCV
    image = cv2.imread(file_path, 0)
    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Get the selected bit plane
    bit_plane = int(bits_var.get())
    print(bit_plane)

    # # Determine the bit depth sequence (8 to 1 bits in 1-bit steps)
    # bit_depth_sequence = range(8, 0, -1)
    #
    # # Process the image for each bit depth and display the results
    # for num_bits in bit_depth_sequence:
    #     # Reduce the gray level resolution
    reduced_resolution_image = reduce_gray_resolution(image, bit_plane)

    # Convert images to PIL format for displaying in the GUI
    original_image = ImageTk.PhotoImage(Image.fromarray(image))
    processed_image = ImageTk.PhotoImage(Image.fromarray(reduced_resolution_image))

    # Display the images in the GUI
    original_image_label.config(image=original_image)
    processed_image_label.config(image=processed_image)
    original_image_label.image = original_image
    processed_image_label.image = processed_image

    # Update the GUI to show the processed image for each bit depth
    root.update_idletasks()
    root.update()


# Function to open an image file and display it
def process_image():
    global original_image, processed_image_label
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load the image using OpenCV
    # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Display the original image
    image = cv2.imread(file_path, 0)

    if image is None:
        print("Error: Could not open or find the image.")
        return
    # Get the selected interpolation method
    interpolation_method = interpolation_var.get()

    # Adjust the bit depth of the processed image
    # max_val = 2 ** bit_depth - 1
    # zoomed_image = (max_val * (zoomed_image / 255)).astype(np.uint8)

    # Define the zoom factor for height (downsampling to 32 pixels in height)
    zoom_factor_height = 32 / image.shape[0]  # Scaling factor for height
    zoom_factor_width = 32 / image.shape[1]  # Scaling factor for width
    # img will be a numpy array of the above shape
    print("image array = ", image)
    # Process the image based on the selected interpolation method
    if interpolation_method == "Nearest Neighbor":
        zoomed_image = nearest_neighbor_interpolation(image, zoom_factor_height, zoom_factor_width)
        zoomed_image = nearest_neighbor_interpolation(zoomed_image, 1 / zoom_factor_height, 1 / zoom_factor_width)
    elif interpolation_method == "Bilinear":
        zoomed_image = bilinear_interpolation(image, zoom_factor_height, zoom_factor_width)
        zoomed_image = bilinear_interpolation(zoomed_image, 1 / zoom_factor_height, 1 / zoom_factor_width)
        # print(zoomed_image)
        # print(zoomed_image.shape)

    elif interpolation_method == "Linear":
        zoomed_image = linear_interpolation(image, zoom_factor_height)
        zoomed_image = linear_interpolation(zoomed_image, 1 / zoom_factor_height)
        print("zoomed_image shape:", zoomed_image.shape)
    # GUI
    # Define the zoom factor for width and height (downsampling to 32x32 pixels)
    # zoom_factor_width = 32 / image.shape[1]  # Scaling factor for width
    # zoom_factor_height = 32 / image.shape[0]  # Scaling factor for height

    # zoomed_image = nearest_neighbor_interpolation(image, zoom_factor_height, zoom_factor_width)
    # resize image
    # zoomed_image = nearest_neighbor_interpolation(zoomed_image, 1 / zoom_factor_height, 1 / zoom_factor_width)

    # Convert images to PIL format for displaying in the GUI
    original_image = ImageTk.PhotoImage(Image.fromarray(image))
    processed_image = ImageTk.PhotoImage(Image.fromarray(zoomed_image))

    # Display the images in the GUI
    original_image_label.config(image=original_image)
    processed_image_label.config(image=processed_image)
    original_image_label.image = original_image
    processed_image_label.image = processed_image


# Create the main application window
root = tk.Tk()
root.title("Image Zooming with Linear Interpolation")

# Create a button to open an image
# open_button = tk.Button(root, text="Open Image", command=open_image)
# open_button.pack(pady=10)

# Create a button to open an image and process it using bit plane slicing
bit_plane_button = tk.Button(root, text="Process Image - Bit Plane Slicing", command=process_bit_plane_slicing)
bit_plane_button.pack(pady=10)

# Create a button to open an image and process it
process_button = tk.Button(root, text="Process Image - Interpolation", command=process_image)
process_button.pack(pady=10)

# Create radio buttons for interpolation selection
interpolation_var = tk.StringVar(value="Nearest Neighbor")

neighbour_button = tk.Radiobutton(root, text="Nearest Neighbor", variable=interpolation_var, value="Nearest Neighbor")
neighbour_button.pack(anchor=tk.W)

linear_button = tk.Radiobutton(root, text="Linear", variable=interpolation_var, value="Linear")
linear_button.pack(anchor=tk.W)

bilinear_button = tk.Radiobutton(root, text="Bilinear", variable=interpolation_var, value="Bilinear")
bilinear_button.pack(anchor=tk.W)

# Create a drop-down menu for selecting bit depth
bits_var = tk.StringVar(value="8")  # Default to 8 bits

bits_label = tk.Label(root, text="Select number of bits")
bits_label.pack(anchor=tk.W)

bits_menu = tk.OptionMenu(root, bits_var, *["1", "2", "3", "4", "5", "6", "7", "8"])
bits_menu.pack(anchor=tk.W)

# Create labels to display the original and processed images
original_image_label = tk.Label(root, text="Original Image")
original_image_label.pack(side=tk.LEFT, padx=10, pady=10)

processed_image_label = tk.Label(root, text="Processed Image")
processed_image_label.pack(side=tk.RIGHT, padx=10, pady=10)

# Run the main event loop
root.mainloop()

# commented out here
# # Display the original image
# img = cv2.imread('Original_lena512.jpg', 0)
#
# # shape prints the tuple (height,weight,channels)
# print("image shape = ", img.shape)
#
# # img will be a numpy array of the above shape
# print("image array = ", img)
#
# # print(img) numpy array is already stored and called img
# print("pixel at index (5,5): ", img[5][5])  # here we are retrieving the value at a specific index
#
# # inspecting img variable
# # print("zoomed_image shape:", zoomed_image.shape)
# # print("zoomed_image type:", type(zoomed_image))
#
# # display image
# cv2.imshow("lena", img)
#
# # waits for user to press any key
# # (this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0)
#
# # closing all open windows
# cv2.destroyAllWindows()
#
# # os.system("pause")
#
# print('Original Dimensions : ', img.shape)
#
# # scale_percent = 60  # percent of original size
# # width = int(img.shape[1] * scale_percent / 100)
# # height = int(img.shape[0] * scale_percent / 100)
# # dim = (width, height)
#
# # resize image
# # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
# # Define the zoom factor for width and height (downsampling to 32x32 pixels)
# zoom_factor_width = 32 / img.shape[1]  # Scaling factor for width
# zoom_factor_height = 32 / img.shape[0]  # Scaling factor for height
#
# resized = nearest_neighbor_interpolation(img, zoom_factor_height, zoom_factor_width)
#
# # display image
# print('Resized Dimensions : ', resized.shape)
# # cv2.imshow("lena resized", resized)
#
# resized = nearest_neighbor_interpolation(resized, 1 / zoom_factor_height, 1 / zoom_factor_width)
#
# cv2.imshow("lena resized", resized)
# # waits for user to press any key
# # (this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0)
#
# # closing all open windows
# cv2.destroyAllWindows()
# end comments

