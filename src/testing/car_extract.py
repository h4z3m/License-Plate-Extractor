import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
from skimage.segmentation import clear_border


minAR=2
maxAR=4
minArea = 600
maxArea = 3500

def plot_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)
    
    

def niblack_thresholding(image, window_size, k):
    mean = cv2.boxFilter(image, cv2.CV_32F, (window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.sqrBoxFilter(image, cv2.CV_32F, (window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    variance = sqmean - (mean ** 2)
    stddev = np.sqrt(variance)
    threshold = mean + k * stddev
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    niblack = (image > threshold).astype(np.uint8) * 255
    return niblack
    

image = cv2.imread('Vehicles/0003.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plot_images(image,gray,"vechile RGB","vechile GRAY")

gaussian_filtered_image = cv2.GaussianBlur(gray, (5, 5), 0)

plot_images(gray,gaussian_filtered_image,"vechile gray","vechile gaussian")

niblack_thresholded = niblack_thresholding(gaussian_filtered_image, window_size=35, k=-0.5)

plot_images(gaussian_filtered_image,niblack_thresholded,"vechile gaussian","niblack_thresholded")

ret, otsu_thresh = cv2.threshold(gaussian_filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plot_images(gaussian_filtered_image,otsu_thresh,"vechile gaussian","otsu_thresh")



# Define a structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 65))

tophat = cv2.morphologyEx(niblack_thresholded, cv2.MORPH_TOPHAT, kernel)
plot_images(otsu_thresh,tophat,"otsu_thresh","tophat")

_, mask = cv2.threshold(tophat, 100, 255, cv2.THRESH_BINARY)
plot_images(otsu_thresh,mask,"otsu_thresh","mask")

cars = cv2.bitwise_and(image, image, mask=mask)

plot_images(image,cars,"otsu_thresh","cars")
"""
sobelx = cv2.Sobel(cars, cv2.CV_64F, 1, 0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(cars, cv2.CV_64F, 0, 1, ksize=5)  # Sobel Edge Detection on the Y axis

# Combined X and Y Sobel Edge Detection
sobel_combined = cv2.bitwise_or(sobelx, sobely)
plot_images(edges,cars,"sobel","cars")
"""

blurred_cars = cv2.GaussianBlur(cars, (9, 9), 0)

edges = cv2.Canny(blurred_cars, 80,120 )

plot_images(cars,edges,"cars","canny_edges")
"""
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
plot_images(edges,closed_image,"edges","closed_edges")
"""
"""
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image=image.copy()
"""
kernel = np.ones((5,5),np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations = 1)
eroded_edges=dilated_edges
plot_images(eroded_edges,edges,"dilated_edges","canny_edges")



contours, hierarchy = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image=image.copy()
if contours:
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Cut out the region of interest from the original image
    cut_out = image[y:y+h, x:x+w]
   


plot_images(image,cut_out,"image in","cut image")

################################image cut#######################################
"""
image = cut_out
gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
gray=clahe.apply(gray1)

plot_images(gray1,gray,"gray","after histogramequalization")
 

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 120, 255,cv2.THRESH_BINARY)[1]

plot_images(gray,light,"gray","light")


rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
plot_images(image,blackhat,"image","mask")

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
#gradX= cv2.Canny(blackhat, 60,120 )

plot_images(gray,gradX,"gray","edged")


gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE,rectKern)
thresh = cv2.threshold(gradX,   0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

plot_images(gray,thresh,"gray","thresholded")


thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

plot_images(gray,thresh,"gray","thresholded after erode dilate")

thresh = cv2.bitwise_and(thresh, thresh, mask=light)
thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.erode(thresh, None, iterations=1)

plot_images(gray,thresh,"gray","After anding with light and closing")



cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

lpCnt = None
roi = None

for c in cnts:
   #epsilon = 0.05 * cv2.arcLength(c, True)
   #approx = cv2.approxPolyDP(c, epsilon, True)

   #if len(approx) == 4:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if minAR <= ar <= maxAR and h > 10 and minArea <= cv2.contourArea(c) <= maxArea:
        lpCnt = c
        licensePlate = gray[y:y + h, x:x + w+40]
        roi = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        break

plot_images(image,licensePlate)
roi=clear_border(roi)
plot_images(image,roi)

"""





















