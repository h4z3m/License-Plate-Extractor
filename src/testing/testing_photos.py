import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from skimage.segmentation import clear_border


minAR=1.2
maxAR=4.3
minArea = 600
maxArea = 8000


def plot_images(img1, img2, title1="Original", title2="Gray"):
    plt.figure(figsize=[7,7])

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = original image
    plt.imshow(img1, cmap="gray")
    plt.title(title1)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = gray image
    plt.imshow(img2, cmap="gray")
    plt.title(title2)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()

def draw_contours(thresholded,image):
    
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image
    

for i in range(1,20):
    filename = 'Vehicles/{:04d}.jpg'.format(i)
#filename = 'Vehicles/0004.jpg'
    image = cv2.imread(filename)
    image = cv2.bilateralFilter(image, d=5, sigmaColor=80, sigmaSpace=80)
    #resize image to 1024 width
    # scale_factor = 1024 / image.shape[1]
    # image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    gray_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(15, 10))
    gray=clahe.apply(gray_1)

    #plot_images(gray_1,gray,"gray","after histogramequalization")
        
    radius = 15
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1), (radius, radius))
    opened_image = cv2.morphologyEx(gray, cv2.MORPH_OPEN, disk_kernel)

    subtracted_image = cv2.subtract(gray, opened_image)
    #plot_images(gray,subtracted_image,"gray","Remainder of opening")
    """ perform a blackhat morphological operation to reveal dark characters (letters, digits, and symbols)
        against light backgrounds (the license plate itself)"""
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    #plot_images(image,blackhat,"image","blackhat")



    """ find regions in the image that are light and may contain license plate characters
    find regions in the image that are light and may contain license plate characters """
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    light = cv2.morphologyEx(subtracted_image, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #plot_images(gray,light,"gray","white in image")


    """ detect edges in the image and emphasize the boundaries of 
    the characters in the license plate"""
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
    #gradX= cv2.Canny(blackhat, 20,130 )
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    #plot_images(gray,gradX,"gray"," x-axis edges")
    """smooth to group the regions that may contain boundaries to license plate characters"""

    gradX = cv2.GaussianBlur(gradX, (7,5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE,rectKern)
    thresh = cv2.threshold(gradX,0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #plot_images(gray,thresh,"gray","thresholded")

    """ there are many other large white regions as well  
        perform a series of erosions and dilations in an attempt to denoise
    """
    thresh = cv2.erode(thresh, None, iterations=2)   #Ziad: iterations=5
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    #plot_images(gray,thresh,"gray","thresholded after erode dilate")

    """
    light image serves as our mask for a bitwise-AND between the thresholded result and 
    the light regions of the image to reveal the license plate candidates.
    follow with a couple of dilations and an erosion 
    to fill holes and clean up the image
        """
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    #plot_images(gray,thresh,"gray","After anding with light and closing")

    """ 
        show me all contours
    """
    # all_Contours=draw_contours(thresh.copy(),image.copy())
    # plot_images(image,all_Contours,"image","all contours")

    """
    find more relative contour

    """
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    contour_image = image.copy()
    #cv2.drawContours(contour_image, cnts, -1, (255, 0, 0), 3)
    #plot_images(image, contour_image, "image", "all contours")



    lpCnt = None
    roi = None
    main_contours = [] # list of contours that match the aspect ratio
    for c in cnts:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        area_ratio = area / (w*h)
        ar = w / float(h)
        if minAR <= ar <= maxAR and 10<= h <=54  and minArea <= w*h <= maxArea:
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            #roi = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            main_contours.append(c)
    #main_contours = sorted(main_contours, key=cv2.contourArea, reverse=True)[:5]
    #print("main_contours size",len(main_contours))
    img_copy = image.copy()
    for c in main_contours:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plot_images(image,contour_image,"Original Image", "All Contours")
    plot_images(image, img_copy, "Original Image", "Contours matching the aspect ratio")
    ######################################################################
    most_rectangular_contour = None
    min_diff = float('inf')
    last_contour = []
    x_final = 0
    y_final = 0
    w_final = 0
    h_final = 0
    # Iterate over the contours
    for c in main_contours:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        # Calculate the aspect ratio of the bounding rectangle
        ar = w / float(h)
        # Calculate the difference between the aspect ratio of the bounding rectangle and the aspect ratio of a perfect rectangle
        diff = abs(ar -3.5)
        # If this difference is smaller than the smallest difference we've seen so far, update the most rectangular contour and the smallest difference
        if diff < min_diff:
            most_rectangular_contour = c
            min_diff = diff
            x_final = x
            y_final = y
            w_final = w
            h_final = h
    last_contour.append(most_rectangular_contour)
    licensePlate = gray[y_final:y_final + h_final, x_final:x_final + w_final]
    # Now most_rectangular_contour is the contour with the most rectangular shape
    ######################################################################
    if (licensePlate != None).any():
        plot_images(image,licensePlate,"image","licensePlate")
##roi=clear_border(roi)
     
    
            
