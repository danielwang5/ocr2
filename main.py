import numpy as np
import cv2
from collections import defaultdict

img = cv2.imread('example_01.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,127,255,0)
img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda contours: get_roi_area(contours)[1])
contours = sorted_contours[1:]

levels = defaultdict(list)
for i in range(len(contours)-1):
	for j in range(i, len(contours)):
		contour1, contour2 = contours[i], contours[j]
		if inside(contour1,contour2):
			levels[i].append(j)
		elif inside(contour2,contour1):
			levels[j].append(i)

out_contours = list(levels.keys())



def get_roi_area(contour):
	x, y, w, h = cv2.boundingRect(contour)
	roi = img[y:y + h, x:x + w]
	area = w*h
	return roi, area

def inside(contour1,contour2):
	xs1, ys1, w1, h1 = cv2.boundingRect(contour1)
	xe1, ye1 = xs1 + w1, ys1 + h1
	xs2, ys2, w2, h2 = cv2.boundingRect(contour2)
	xe2, ye2 = xs2 + w2, ys2 + h2
	return (xs1 < xs2) and (ys1 < ys2) and (xe1 > xe2) and (ye1 > ye2)



for i, contour in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(contour)

    roi = img[y:y + h, x:x + w]
    area = w*h

    if 250 < area:
        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('rect', rect)

cv2.waitKey(10000)


