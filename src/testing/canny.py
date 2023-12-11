import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from skimage.segmentation import clear_border


minAR=1.5
maxAR=3
minArea = 300
maxArea = 3500


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
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    # Draw all contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image
    



image = cv2.imread('Vehicles/0009.jpg')
gray_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
gray=clahe.apply(gray_1)

plot_images(gray_1,gray,"gray","after histogramequalization")
 


""" perform a blackhat morphological operation to reveal dark characters (letters, digits, and symbols)
 against light backgrounds (the license plate itself)"""
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
plot_images(image,blackhat,"image","blackhat")



""" find regions in the image that are light and may contain license plate characters
find regions in the image that are light and may contain license plate characters """
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 120, 255,cv2.THRESH_BINARY)[1]
plot_images(gray,light,"gray","white in image")


""" detect edges in the image and emphasize the boundaries of 
the characters in the license plate"""
"""
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
gradY=cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=0, dy=1, ksize=-1)
gradX=cv2.bitwise_or(gradX,gradY)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
"""
gradX= cv2.Canny(blackhat, 0,255 )
canny=gradX.copy()
plot_images(gray,gradX,"gray"," edges")
"""smooth to group the regions that may contain boundaries to license plate characters"""

gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
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
all_Contours=draw_contours(canny.copy(),image.copy())
plot_images(image,all_Contours,"image","all contours")

"""
find more relative contour

"""
cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[2]*cv2.boundingRect(x)[3], reverse=True)[:10]

lpCnt = None
roi = None

for i,c in enumerate(cnts) :
   #epsilon = 0.05 * cv2.arcLength(c, True)
   #approx = cv2.approxPolyDP(c, epsilon, True)
   
   image_copy=image.copy()
   #if len(approx) == 4:
   print(cv2.contourArea(c))
   if cv2.contourArea(c)>35:
       peri=cv2.arcLength(c, True)
       approx=cv2.approxPolyDP(c,0.04*peri,True)
       if cv2.contourArea(c)>35 :
           cv2.drawContours(image_copy, cnts, i, (0, 255, 0), 3)
           (x, y, w, h) = cv2.boundingRect(c)
           ar = w / float(h)
           if minAR <= ar <= maxAR and w>h :
               plot_images(image,image_copy,str(len(approx)),str(i))

               

           
           
    
    
   """
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if minAR <= ar <= maxAR and minArea <= cv2.contourArea(c) <= maxArea:
        lpCnt = c
        licensePlate = gray[y:y + h, x:x + w]
        roi = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        break
   """
##plot_images(image,licensePlate,"image","licensePlate")
##roi=clear_border(roi)
##plot_images(image,roi)

        
