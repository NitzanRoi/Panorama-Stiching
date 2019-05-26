
from skimage import color
from skimage import io
import numpy as np
from scipy import signal

GREYSCALE_CODE = 1
MAX_PIXEL_VAL = 255
INVALID_FILTER_SIZE_MSG = "Error: the given size of the filter is not valid"
GAUSSIAN_BASIC_KERNEL = np.array([1, 1])
MIN_DIM_GAUSSIAN = 16
INVALID_BLUR_KERNEL_MSG = "Error: the given kernel size is not valid"

def read_image(filename, representation):
    """
    this function reads an image file and returns it in a given representation
    filename is the image
    representation code: 1 is greyscale, 2 is RGB
    returns an image
    """
    final_img = io.imread(filename).astype(np.float64)
    if (representation == GREYSCALE_CODE):
        final_img = color.rgb2gray(final_img)
    final_img /= MAX_PIXEL_VAL
    return final_img.astype(np.float64)

def is_even_number(num):
    """
    return boolean - if num is even number
    """
    return (num % 2 == 0)

# from ex3:

def create_gaussian(kernel_size):
    """
    calculates the gaussian using binomial coefficients and returns it
    """
    gaussian = GAUSSIAN_BASIC_KERNEL.copy()
    for i in range(kernel_size - 2):
        gaussian = signal.convolve(gaussian, GAUSSIAN_BASIC_KERNEL)
    if (np.sum(gaussian) != 0):
        return gaussian / np.sum(gaussian)
    return gaussian

def blur(im, kernel):
    """
    performs image blurring using convolution between image and Gaussian filter
    (once as a row vector and once as a column vector)
        im - the image, float64 greyscale
        kernel - the gaussian kernel
    returns output as float64 greyscale
    """
    row_blur = signal.convolve2d(im, kernel, mode='same').astype(np.float64)
    return signal.convolve2d(row_blur, kernel.T, mode='same').astype(np.float64)

def get_even_idx_values(row):
    """
    given row - returns its values which are in the even indexes
    """
    even_indexes = np.arange(0, row.shape[0], 2)
    return np.take(row, even_indexes)

def down_sample(image):
    """
    returns down-sampled image (i.e. reduce function)
    """
    return np.apply_along_axis(get_even_idx_values, 1, image[::2])

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    constructs a Gaussian pyramid with the given image
        im - a grayscale float64 double image, values in range [0, 1]
        max_levels - maximal number of levels in the resulting pyramid
                    (including the given image)
        filter_size - the size of the filter, an odd number
    returns:
        pyr - standard *python* array with max length of max_levels parameter
              each element is a greyscale image
        filter_vec - a row vector of shape (1, filter_size), normalized
    """
    if (filter_size <= 1 or is_even_number(filter_size)):
        print(INVALID_FILTER_SIZE_MSG)
        exit()
    pyr = [im]
    filter_vec = create_gaussian(filter_size).reshape(1, filter_size)
    for i in range(max_levels - 1):
        im = blur(im, filter_vec)
        im = down_sample(im)
        minimum_stop_condition = min(im.shape[0], im.shape[1])
        if (minimum_stop_condition < MIN_DIM_GAUSSIAN):
            break
        pyr.append(im)
    return pyr, filter_vec

# from ex2:

def create_gaussian_kernel(kernel_size):
    """
    calculates the gaussian 2D kernel using binomial coefficients and returns it
    """
    gaussian_x = np.zeros((kernel_size, kernel_size))
    gaussian_y = np.zeros((kernel_size, kernel_size))
    binomial_coefficients = GAUSSIAN_BASIC_KERNEL.copy()
    for i in range(kernel_size - 2):
        binomial_coefficients = signal.convolve(binomial_coefficients, GAUSSIAN_BASIC_KERNEL)
    gaussian_x[int(np.floor(kernel_size / 2)), :] = binomial_coefficients
    gaussian_y[:, int(np.floor(kernel_size / 2))] = binomial_coefficients
    gaussian_kernel = signal.convolve2d(gaussian_x, gaussian_y, mode='same')
    if (np.sum(gaussian_kernel) != 0):
        return gaussian_kernel / np.sum(gaussian_kernel)
    return gaussian_kernel

def blur_spatial(im, kernel_size):
    """
    performs image blurring using 2D convolution between image and Gaussian
        im - the image, float64 greyscale
        kernel_size - an odd integer
    returns output as float64 greyscale
    """
    if (kernel_size == 1):
        return im
    if (is_even_number(kernel_size) or kernel_size < 0):
        print(INVALID_BLUR_KERNEL_MSG)
        exit()
    gaussian_kernel = create_gaussian_kernel(kernel_size)
    return signal.convolve2d(im, gaussian_kernel, mode='same').astype(np.float64)