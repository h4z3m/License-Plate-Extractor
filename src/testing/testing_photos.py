import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from skimage.segmentation import clear_border


minAR=1.5
maxAR=4.3
minArea = 600
maxArea = 8000


def plot_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)

def draw_contours(thresholded,image):
    
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image
    

   
for i in range(1,13):
    filename = 'Vehicles/{:04d}.jpg'.format(i)
    
    image = cv2.imread(filename)
    image = cv2.bilateralFilter(image, d=5, sigmaColor=80, sigmaSpace=80)
    gray_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(15, 10))
    gray=clahe.apply(gray_1)
    
    plot_images(gray_1,gray,"gray","after histogramequalization")
     
    radius = 12
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1), (radius, radius))
    opened_image = cv2.morphologyEx(gray, cv2.MORPH_OPEN, disk_kernel)
    plot_images(opened_image,gray,"opened","gray")
    subtracted_image = cv2.subtract(gray, opened_image)
    
    """ perform a blackhat morphological operation to reveal dark characters (letters, digits, and symbols)
     against light backgrounds (the license plate itself)"""
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    plot_images(image,blackhat,"image","blackhat")
    
    
    
    """ find regions in the image that are light and may contain license plate characters
    find regions in the image that are light and may contain license plate characters """
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(subtracted_image, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    plot_images(gray,light,"gray","white in image")
    
    
    """ detect edges in the image and emphasize the boundaries of 
    the characters in the license plate"""
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    #gradX= cv2.Canny(blackhat, 60,120 )
    
    
    plot_images(gray,gradX,"gray"," x-axis edges")
    """smooth to group the regions that may contain boundaries to license plate characters"""
    
    gradX = cv2.GaussianBlur(gradX, (7, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE,rectKern)
    thresh = cv2.threshold(gradX,   0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    plot_images(gray,thresh,"gray","thresholded")
    
    """ there are many other large white regions as well  
        perform a series of erosions and dilations in an attempt to denoise
    """
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    plot_images(gray,thresh,"gray","thresholded after erode dilate")
    
    """
    light image serves as our mask for a bitwise-AND between the thresholded result and 
    the light regions of the image to reveal the license plate candidates.
    follow with a couple of dilations and an erosion 
    to fill holes and clean up the image
     """
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    
    plot_images(gray,thresh,"gray","After anding with light and closing")
    
    """ 
     show me all contours
    """
    all_Contours=draw_contours(thresh.copy(),image.copy())
    plot_images(image,all_Contours,"image","all contours")
    
    """
    find more relative contour
    
    """
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    
    lpCnt = None
    roi = None
    
    for c in cnts:
       #epsilon = 0.05 * cv2.arcLength(c, True)
       #approx = cv2.approxPolyDP(c, epsilon, True)
    
       #if len(approx) == 4:
        area = cv2.contourArea(c)

        (x, y, w, h) = cv2.boundingRect(c)
        area_ratio = area / (w*h)
        ar = w / float(h)
        if minAR <= ar <= maxAR and 10<= h <=54  and minArea <= w*h <= maxArea:
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            break
   
    plot_images(image,image,"image","licensePlate")

    if (licensePlate != None).any():
        plot_images(image,licensePlate,"image","licensePlate")
    ##roi=clear_border(roi)
     
    
            
