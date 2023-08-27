import numpy as np
import cv2
import copy

def get_len_square_side_px(img): # original image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_len=-1
    blended = copy.deepcopy(img)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 1000: 
            # using boundingRect to find the correct box
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            len_side = (w+h)/2
            if ratio >= 0.8 and ratio <= 1.2 and len_side>max_len:
                max_len = len_side
                approx1 = approx
                # using arcLength to calculate the area
                side_length_in_pixels = cv2.arcLength(cnt, True)/4
    if max_len!=-1:
        blended = cv2.drawContours(blended, [approx1], -1, (0,255,255), 5)
        return blended, side_length_in_pixels #in pixels
    else:
        print("Error: Square not found ", max_len)
        return blended, -1





def get_area_and_perim_px(img): # gray image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
    wound_area = 0
    wound_perimeter = 0
    for num, cnt in enumerate(contours):
        #calculate area in pixels
        contour_area = cv2.contourArea(cnt)
        if hierarchy[0, num][3] != -1:
            wound_area -= contour_area
        else:
            wound_area += contour_area
        #calculate perimeter in pixels
        wound_perimeter += cv2.arcLength(cnt, True)
        
    return wound_area, wound_perimeter # in pixels





side_length_square_in_mm = 20

def get_perim_mm(perim_in_px, len_side): # in px
    side_length_square_in_px = len_side
    real_length_per_px = side_length_square_in_mm / side_length_square_in_px
    perim = real_length_per_px * perim_in_px
    return perim # in mm





area_square_in_mm = side_length_square_in_mm**2

def get_area_mm(area_in_px, len_side): #in px
    area_square_in_px = len_side*len_side
    real_area_per_px = area_square_in_mm / area_square_in_px
    # object area in mm^2
    area_in_mm = real_area_per_px * area_in_px
    return area_in_mm, real_area_per_px # in mm^2



def get_area_and_perim_in_mm(img, masks): # original image and mask
    darkImg = np.zeros((img.shape[0], img.shape[1]))
    for mask in masks:
        darkImg[mask['mask']==True] = 255
    darkImg = darkImg.astype(np.uint8)
    # get length of square side in pixels
    processed_image, len_side = get_len_square_side_px(img)
    # calculate area and perimeter in pixels
    area_px, perim_px = get_area_and_perim_px(darkImg)
    area_mm, real_area_per_px = get_area_mm(area_px, len_side)
    perim_mm = get_perim_mm(perim_px, len_side)

    return processed_image, area_mm, perim_mm, real_area_per_px






