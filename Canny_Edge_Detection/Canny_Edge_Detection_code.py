import numpy as np
from PIL import Image

def convert_to_grayscale(image):
  grayscale_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      r, g, b = image[x][y]
      #weighted contributions of red, green and blue channels
      gray_value = 0.299 * r + 0.587 * g + 0.114 * b
      grayscale_image[x][y] = gray_value
  return grayscale_image

'''
K(x, y) = (1 / (2 * pi * sigma^2)) * exp(-((i-k-1)^2 + (j-k-1)^2) / (2 * sigma^2)), where i,j range from 1 to size,
and k = size//2
'''
def gaussian_kernel(size, sigma=1):
  x, y = np.mgrid[1:size+1, 1:size+1]
  k = size//2
  normal = 1 / (2.0 * np.pi * sigma**2)
  expterm = np.exp(-(((x-k-1)**2 + (y-k-1)**2) / (2.0*sigma**2)))
  gaussian = normal * expterm
  return gaussian

'''
For output to be in same dimension as input (i.e. same convolution), need to pad the image
Here pad the image with 0s
The size of padding is given by (k-1)/2 where k is the size of kernel
'''
def convolution(image, kernel):
  image_row, image_col = image.shape
  kernel_row, kernel_col = kernel.shape

  output = np.zeros(image.shape)

  pad_height = int((kernel_row - 1) / 2)
  pad_width = int((kernel_col - 1) / 2)
  padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
  padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

  for row in range(image_row):
    for col in range(image_col):
        output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

  return output

def sobel_edge_detection(image, filter, convert_to_degree=False):
  new_image_x = convolution(image, filter)
  new_image_y = convolution(image, np.flip(filter.T, axis=0))
  #combine both into 1 gradient image using sqrt of Gx**2 + Gy**2
  gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
  #normalize the output to be between 0 and 255
  gradient_magnitude *= 255.0 / gradient_magnitude.max()
  #np.arctan2 returns angles in radians
  gradient_direction = np.arctan2(new_image_y, new_image_x)

  if convert_to_degree:
    #convert radians to degrees
    gradient_direction = gradient_direction * 180 / np.pi
    #add 180 to the negative angles so that all angles are between 0 and 180
    gradient_direction[gradient_direction < 0] += 180

  return gradient_magnitude, gradient_direction

'''
remove the redudant/duplicate edges identified by Sobel filters using Non-Max Suppression Algorithm
'''
def non_max_suppression(gradient_magnitude, gradient_direction):
  image_row, image_col = gradient_magnitude.shape
  output = np.zeros(gradient_magnitude.shape)
  #set to PI to 180 degrees
  PI = 180
  #loop through all the pixels in the gradient directions ( except the border pixels )
  for row in range(1, image_row - 1):
    for col in range(1, image_col - 1):
        direction = gradient_direction[row, col]
        #based on the value of gradient direction store the gradient magnitude of the two neighboring pixel.
        if (0 <= direction < PI / 8) or (7 * PI / 8 <= direction <= PI):
            before_pixel = gradient_magnitude[row, col - 1]
            after_pixel = gradient_magnitude[row, col + 1]
        elif (PI / 8 <= direction < 3 * PI / 8):
            before_pixel = gradient_magnitude[row + 1, col - 1]
            after_pixel = gradient_magnitude[row - 1, col + 1]
        elif (3 * PI / 8 <= direction < 5 * PI / 8):
            before_pixel = gradient_magnitude[row - 1, col]
            after_pixel = gradient_magnitude[row + 1, col]
        else:
            before_pixel = gradient_magnitude[row - 1, col - 1]
            after_pixel = gradient_magnitude[row + 1, col + 1]
        # find out whether the selected/middle pixel has the highest gradient magnitude or not
        # if highest, update the output image for the given row and col with the value of the gradient magnitude
        if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
            output[row, col] = gradient_magnitude[row, col]
  return output

'''
there are still some edges between gray to dark-gray
objective is to produce clear edges ( all the edge pixel will be 255 ) using Hysteresis Threshold
has 2 parts: apply thresholding & apply hysteresis

thresholding takes all the edges, defines them as 
weak (say intensity value 25) or strong (intensity value 255) or irrelevant (intensity value 0)
'''
def double_thresholding(image, low_threshold, high_threshold, weak):
  output = np.zeros(image.shape, dtype=np.uint8)
  strong = np.uint8(255)
  strong_row, strong_col = np.where(image >= high_threshold)
  weak_row, weak_col = np.where((image <= high_threshold) & (image >= low_threshold))
  output[strong_row, strong_col] = strong
  output[weak_row, weak_col] = weak
  return output

'''
objective of the hysteresis function is to identify the weak pixels which can be edges and discard the remaining
find out whether a selected weak pixel is connected to the already defined edge pixels
if so can consider this weak pixel also to be part of an edge
so find out whether any of the 8 neighbor pixels has value equal to 255,
if yes then change the value of the weak pixel to 255, otherwise make it irrelevant by changing the value to 0
'''

# def hysteresis_from_source(image, weak):
#     image_row, image_col = image.shape 
#     top_to_bottom = image.copy() 
#     for row in range(1, image_row):
#         for col in range(1, image_col):
#             if top_to_bottom[row, col] == weak:
#                 if (top_to_bottom[row, col + 1] == 255 or 
#                 top_to_bottom[row, col - 1] == 255 or 
#                 top_to_bottom[row - 1, col] == 255 or 
#                 top_to_bottom[row + 1, col] == 255 or 
#                 top_to_bottom[row - 1, col - 1] == 255 or 
#                 top_to_bottom[row + 1, col - 1] == 255 or 
#                 top_to_bottom[row - 1, col + 1] == 255 or 
#                 top_to_bottom[row + 1, col + 1] == 255):
#                     top_to_bottom[row, col] = 255
#                 else:
#                     top_to_bottom[row, col] = 0

def hysteresis(image, weak, row_loop, col_loop):
  row_start, row_end, row_step = row_loop
  col_start, col_end, col_step = col_loop
  for row in range(row_start, row_end, row_step):
    for col in range(col_start, col_end, col_step):
      if image[row, col] == weak:
        if (image[row, col + 1] == 255 or
            image[row, col - 1] == 255 or
            image[row - 1, col] == 255 or
            image[row + 1, col] == 255 or
            image[row - 1, col - 1] == 255 or
            image[row + 1, col - 1] == 255 or
            image[row - 1, col + 1] == 255 or
            image[row + 1, col + 1] == 255):
          image[row, col] = 255
        else:
          image[row, col] = 0

'''
hysteresis from all 4 corners of the image to avoid edge discontinuities
'''
def get_final_edge_image(image, weak):
  image_row, image_col = image.shape
  #hysteresis while scanning the image from top-left to bottom-right corner
  top_to_bottom = image.copy()
  hysteresis(top_to_bottom, weak, (1, image_row, 1), (1, image_col, 1))
  #hysteresis while scanning the image from bottom-right to top-left corner
  bottom_to_top = image.copy()
  hysteresis(bottom_to_top, weak, (image_row - 1, 0, -1), (image_col - 1, 0, -1))
  #hysteresis while scanning the image from top-right to bottom-left corner
  right_to_left = image.copy()
  hysteresis(right_to_left, weak, (1, image_row, 1), (image_col - 1, 0, -1))
  #hysteresis while scanning the image from bottom-left to top-right corner
  left_to_right = image.copy()
  hysteresis(left_to_right, weak, (image_row - 1, 0, -1), (1, image_col, 1))

  #Sum all the pixels to create final image and threshold to 255
  final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
  final_image[final_image > 255] = 255
  return final_image

def canny_edge_detection(image_path, save_path, low_threshold, high_threshold, gaussian_kernel_size):
  img = Image.open(image_path)
  img = np.array(img)
  #convert to grayscale if RGB image
  if len(img.shape) == 3 and img.shape[2] == 3:
    img = convert_to_grayscale(img)
  gaus_kernel = gaussian_kernel(gaussian_kernel_size)
  gaus_filtered_img = convolution(img, gaus_kernel)
  sobel_filter_mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  grad_mag, grad_dir = sobel_edge_detection(gaus_filtered_img, sobel_filter_mx, convert_to_degree=True)
  non_max_suppressed_img = non_max_suppression(grad_mag, grad_dir)
  weak = np.uint8(25) #intensity value chosen for weak pixels
  thresholded_img = double_thresholding(non_max_suppressed_img, low_threshold, high_threshold, weak)
  edge_img = get_final_edge_image(thresholded_img, weak)
  Image.fromarray(edge_img).save(save_path)

canny_edge_detection('lena.png', 'edge_image.png', 10, 25, 5)