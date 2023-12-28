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
  #ensure that the value is in the valid range [0, 255] for an 8-bit image
  return grayscale_image.astype(np.uint8)

def resample_image(image, consider_scale_factor, scale_factor=1, new_height=200, new_width=200):
  # if scaling factor is to be considered
  if consider_scale_factor:
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
  # if the newly given height and weight dimensions are to be considered
  else:
    scale_factor = new_height/image.shape[0]

  resampled_image = np.zeros((new_height, new_width), np.uint8)
  
  for i in range(new_height):
    for j in range(new_width):      
      y = i / scale_factor
      x = j / scale_factor
      #coordinates of the 4 closest points
      xl, yl = int(x), int(y)
      xh, yh = min(int(x)+1, image.shape[1]-1), min(int(y)+1, image.shape[0]-1)

      fxlyl = image[yl][xl]
      fxhyl = image[yl][xh]
      fxlyh = image[yh][xl]
      fxhyh = image[yh][xh]
      
      resampled_image[i][j] = (
          fxlyl * (xh-x) * (yh-y) + 
          fxlyh * (xh-x) * (y-yl) + 
          fxhyl * (x-xl) * (yh-y) + 
          fxhyh * (x-xl) * (y-yl)
      ).astype(np.uint8)

  return resampled_image

def get_mean_squared_difference(image1, image2):
  squared_diff = (image1 - image2)**2
  # Calculate the average squared difference
  average_squared_diff = np.mean(squared_diff)
  return average_squared_diff

if __name__ == "__main__":
  img = Image.open('sunflower.jpg')
  img = np.array(img)

  #save the grayscaled image
  grayscale_img = convert_to_grayscale(img)
  Image.fromarray(grayscale_img).save('grayscale_sunflower.jpg')

  #resample grayscaled image to 0.7 times original dimensions
  resample_img = resample_image(grayscale_img, True, scale_factor=0.7)
  Image.fromarray(resample_img).save('resample_sunflower.jpg')

  #resample above image to original size
  resample_back_img = resample_image(resample_img, False, new_height=img.shape[0], new_width=img.shape[1])
  Image.fromarray(resample_back_img).save('resample_back_sunflower.jpg')

  #compute average squared difference of pixel values
  diff = get_mean_squared_difference(grayscale_img, resample_back_img)
  #write this value to output.txt file
  file_path = 'output.txt'
  with open(file_path, 'w') as file:
    output = f'The average of squared difference between pixels = {diff}'
    file.write(output)