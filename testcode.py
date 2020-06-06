import cv2
import numpy as np
import pytesseract
import preprocess as pre
import imutils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread("testimg/test6.jpg")
copy = img.copy()

gray = pre.get_gray(img)
blur = pre.gaussianblur(gray)
canny = pre.cannyedge(blur)

#cv2.imshow("Original", img)
#cv2.imshow("gray", gray)
#cv2.imshow("Blur", blur)
cv2.imshow("Canny", canny)
edged = canny.copy()

BorderCont = pre.countours(edged)
#print(BorderCont)

bordered = pre.drawborder(img, BorderCont)
cv2.imshow("Bordered", bordered)

BorderCont = BorderCont.reshape(4,2)

rect = np.zeros((4,2), dtype = "float32")

(rect, dst, width, height) = pre.getimgprop(BorderCont)

scan = pre.birdseyeview(copy, rect, dst, width, height)
cv2.imshow("Scan", scan)

final = pre.get_gray(scan)
final = pre.adaptivethreshold(final)
cv2.imshow("Final Scan", final)

text = pytesseract.image_to_string(final)
print(text)
cv2.waitKey(0)
#cv2.destroyAllWindows()
