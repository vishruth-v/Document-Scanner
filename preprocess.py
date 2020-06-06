import cv2
import numpy as np
import imutils

#Funtion to get grayscale image
def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Blur
def gaussianblur(img):
    return cv2.GaussianBlur(img, (5,5), 0)

#Canny Edge detection
def cannyedge(img):
    return cv2.Canny(img, 100, 300, 3)

#Finding biggest Contour
def countours(img):
    allcont = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allcont = imutils.grab_contours(allcont)
    #print(allcont)
    allcont = sorted(allcont, key = cv2.contourArea, reverse= True)[:5]

    for c in allcont:
        peri = cv2.arcLength(c, True)
        dimensions = cv2.approxPolyDP(c, 0.02*peri, True)

        if len(dimensions) == 4:
            BorderCont = dimensions
            break
    
    #peri = cv2.arcLength(allcont[0], True)
    #BorderCont = cv2.approxPolyDP(allcont[0], 0.02*peri, True)
    return BorderCont

#Draw Contour
def drawborder(img, BorderCont):
    copy = img.copy()
    cv2.drawContours(copy, [BorderCont], -1, (0,255,0), 2)
    return copy


#Get properties of the image like width, height and arrange coords in order
def getimgprop(BorderCoords):
    rect = np.zeros((4,2), dtype = "float32") #empty array to put into order
    
    #To identify the corners
    #Based on sum of coords, top-left will have smallest and bottom-right will have largest 
    sums = np.sum(BorderCoords, axis = 1) 
    rect[0] = BorderCoords[np.argmin(sums)]
    rect[2]= BorderCoords[np.argmax(sums)]
    #Based on differnce of coords, top-right will have smallest and bottom-left will have largest 
    diff = np.diff(BorderCoords, axis = 1) 
    rect[1] = BorderCoords[np.argmin(diff)]
    rect[3] = BorderCoords[np.argmax(diff)]

    (tl,tr,br,bl) = rect
    #Compute width of image
    topwidth = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2)
    bottomwidth = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2)
    maxwidth = max(int(topwidth), int(bottomwidth))

    #Compute width of image
    leftheight = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
    rightheight = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
    maxheight = max(int(leftheight), int(rightheight))

    dst = np.array([
        [0,0],
        [maxwidth-1, 0],
        [maxwidth-1, maxheight-1],
        [0, maxheight-1]], dtype="float32")

    return rect,dst, maxwidth,maxheight

#Perform perspective transform
def birdseyeview (img, rect, dst, width, height):
    # compute the perspective transform matrix and then apply it
    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    # transform ROI
    scan = cv2.warpPerspective(img, transformMatrix, (width, height))
    return scan

def adaptivethreshold(img):
    thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 11)
    return thresholded