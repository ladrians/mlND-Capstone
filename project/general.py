'''
Generic functions
'''

import os
import cv2
import numpy as np
import random
import math
import json

from matplotlib import pyplot as plt

#helper function to show several images
def show_imgs(img_arr, cmap=None):
    
    fig, ax = plt.subplots(1, img_arr.shape[0], figsize=(15, 6),
                             subplot_kw={'adjustable': 'box-forced'})

    axoff = np.vectorize(lambda ax:ax.axis('off'))
    axoff(ax)

    for i, img in enumerate(img_arr):
        ax[i].imshow(img, cmap=cmap)

def duplicate_image(img):
    '''
    Duplicates an image
    '''
    return np.copy(img)

def add_text_to_image(img, msg, position, color):
    '''
    Add a text to an image
    '''
    result = duplicate_image(img)
    cv2.putText(result, msg, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return result

def get_median_blur(gray_frame):
    return cv2.medianBlur(gray_frame, 5)

# Flip image

def flip_image(img):
    '''
    return an image flipped horizontally
    '''
    flip_img = cv2.flip(img, 1)
    return flip_img

def crop_image(img, x, w, y, h):
    '''
    Crop an image, using ideas from https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    '''
    crop_img = img[h:img.shape[0]-w, 0:img.shape[1]]
    return crop_img

def image_augment_brightness(img):
    '''
    return an image with augmented brightness.
    Ideas taken from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    image = np.array(image, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    return image

def image_resize(img, rows, columns):
    '''
    return an image rezed using the rows as height and columns as width.
    discarded this option
    https://discussions.udacity.com/t/how-to-optimize-the-model-to-reach-the-goal/491414/6
    check issue https://github.com/keras-team/keras/issues/5298
    #return ktf.image.resize_images(img, (r, c))
    '''
    img_resize = cv2.resize(np.array(img), (columns, rows), interpolation=cv2.INTER_AREA)
    return img_resize

def normalize_image(img):
    '''
    Normalize the current image
    '''
    return img/255.0 - 0.5

def grayscale(img):
    '''
    Applies the Grayscale transform
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image(label, img, cmap=None, time=500):
    '''
    Shows an image
    '''
    ##cv2.imshow(label,img)
    ##cv2.waitKey(time)
    plt.imshow(img, cmap=cmap)

def compare_images(label1, img1, label2, img2):
    '''
    Compare two images
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    #f.tight_layout()
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title(label1, fontsize=20)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title(label2, fontsize=20)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def resize_image(img, height, weigth):
    return cv2.resize(img, (height, weigth), interpolation=cv2.INTER_AREA)

def save_image(path, img):
    '''
    Saves an image in the specified folder
    '''
    plt.imsave(path, img)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2,slope in line:
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_lane_lines2(image, lines, color=[255, 0, 0], thickness=10):
    """
    alternative method to draw a line on an image
    """

    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return weighted_img(image, line_image, α=0.8, β=1., γ=0.)

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    """
    alternative method to draw a line on an image
    """

    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            #format line to be drawn
            x1, y1, x2, y2 = line[0]
            line_coord = np.array([[[x1, y1], [x2, y2]]], dtype=float)
            #cv2.line(line_image, *line,  color, thickness)
            cv2.line(line_image,(x1,y1),(x2,y2),color,thickness)
    return weighted_img(image, line_image, α=0.8, β=1., γ=0.)

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def hough_lines2(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    based on original hough_lines function
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def get_coef(x1, y1, x2, y2):
    """
    get line coefficient based on two points
    """
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b

def get_distance(x1, y1, x2, y2):
    """
    get distance based on two points
    """
    distance = np.sqrt((y2-y1)**2+(x2-x1)**2)
    return distance

def get_line_median(lines):
    """
    base on lines, get the median for the slopes
    """
    if lines is None or len(lines) == 0:
        return None
    return np.median(lines,axis=0)

def slope(x1, y1, x2, y2):
    try:
        return (y1 - y2) / (x1 - x2)
    except:
        return 0
		
def separate_lines(lines):
    right = []
    left = []

    if lines is not None:
        for x1,y1,x2,y2 in lines[:, 0]:
            m = slope(x1,y1,x2,y2)
            if m >= 0:
                right.append([x1,y1,x2,y2,m])
            else:
                left.append([x1,y1,x2,y2,m])
    return left, right

def reject_outliers(data, cutoff, threshold=0.08, lane='left'):
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    try:
        if lane == 'left':
            return data[np.argmin(data,axis=0)[-1]]
        elif lane == 'right':
            return data[np.argmax(data,axis=0)[-1]]
    except:
        return []

def extend_point(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y

def get_main_lines(shape_y1, shape_y2, lines, color=[255, 0, 0], thickness=2):
    """
    based on draw_lines
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.    
    """
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
        
    if lines is None or len(lines) == 0:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1 or y2==y1:
                continue # ignore vertical and horizontal lines
                print("ignore vertical and horizontal lines", x1,y1)
            m, b = get_coef(x1, y1, x2, y2)
            line_length = get_distance(x1, y1, x2, y2) 
            if m < 0:
                left_lines.append((m, b))
                left_weights.append((line_length))
            else:
                right_lines.append((m, b))
                right_weights.append((line_length))
    
    left_lane = get_line_median(left_lines)
    right_lane = get_line_median(right_lines)

    left_line  = get_points(shape_y1, shape_y2, left_lane)
    right_line = get_points(shape_y1, shape_y2, right_lane)    
    
    return left_line, right_line

def get_points(y1, y2, line):
    """
    get a line based on slope (m) and intercept (b) to pixels
    """
    if line is None:
        return ((0, 0), (0, 0))
        #return None
    m, b = line
    x1 = int((y1 - b)/m)
    x2 = int((y2 - b)/m)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def image_undistort(img, mtx, dist):
    '''
    takes an image and returns an undistorted image
    '''
    return cv2.undistort(img, mtx, dist, None, mtx)

def thresholds(image, 
               s_thresh=(80, 255), # 30, 255 
               sx_thresh=(20, 150), 
               sy_thresh = (20, 255), 
               abs_thresh=(0.9, 1.1), # (0, np.pi/2)
               mag_thresh=(0.9, 1.1), # did not work
               v_thresh=(165,255) # (110,255)
              ): 
    '''
    filter different thresholds
    '''
    img = duplicate_image(image)    
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1    

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1        
    
    # Sobel Threshold x gradient
    gray = grayscale(image)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Sobel Threshold y gradient
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    sy_binary = np.zeros_like(scaled_sobel)
    sy_binary[(scaled_sobel >= sy_thresh[0]) & (scaled_sobel <= sy_thresh[1])] = 1    
    
    # Gradient magnitude
    gradmag = np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1        
    
    # absolute value of the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= abs_thresh[0]) & (absgraddir <= abs_thresh[1])] = 1    
        
    # Combine a binary image of all thresholds
    combined_binary = np.zeros_like(sx_binary)
    
    ## Pipeline to test separate filters
    #combined_binary[(sy_binary == 1)] = 1
    #combined_binary[(sx_binary == 1)] = 1
    #combined_binary[(sy_binary == 1) & (sx_binary == 1)] = 1
    #combined_binary[(s_binary == 1)] = 1
    #combined_binary[(v_binary == 1)] = 1
    #combined_binary[(v_binary == 1) | (s_binary == 1)] = 1
    #combined_binary[(dir_binary == 1)] = 1
    #combined_binary[((v_binary == 1) | (s_binary == 1)) & (dir_binary == 1)] = 1
    #combined_binary[(s_binary == 1) & (v_binary == 1) | (sx_binary == 1) & (sy_binary == 1)] = 1
    #combined_binary[(s_binary == 1) & (v_binary == 1) | (sx_binary == 1) | (sy_binary == 1)] = 1

    # Working pipelines
    #combined_binary[((s_binary == 1) | (sx_binary == 1)) & ((mag_binary == 1) | (dir_binary == 1))] = 1 # initial
    #combined_binary[((s_binary == 1) | (sx_binary == 1)) & ((v_binary == 1) | (dir_binary == 1))] = 1  # worked fine
    combined_binary[((s_binary == 1) | (sx_binary == 1) | (sy_binary == 1)) & ((v_binary == 1) | (dir_binary == 1))] = 1  # worked fine
    
    return combined_binary

def warp(img, M, size):
    '''
    Compute and apply perpective transform
    '''
    img_size = (size[1], size[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

