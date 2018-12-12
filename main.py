import numpy as np
import cv2
from collections import defaultdict


def get_roi_area_coords(contour):
	x, y, w, h = cv2.boundingRect(contour)
	roi = img[y:y + h, x:x + w]
	area = w*h
	coords = [x, y, x+w, y+h]
	return roi, area, coords

def inside(contour1,contour2):
	xs1, ys1, w1, h1 = cv2.boundingRect(contour1)
	xe1, ye1 = xs1 + w1, ys1 + h1
	xs2, ys2, w2, h2 = cv2.boundingRect(contour2)
	xe2, ye2 = xs2 + w2, ys2 + h2
	return (xs1 < xs2) and (ys1 < ys2) and (xe1 > xe2) and (ye1 > ye2)


def inside_select(coords,contour):
	xs1, ys1, xe1, ye1 = coords
	xs2, ys2, w2, h2 = cv2.boundingRect(contour2)
	xe2, ye2 = xs2 + w2, ys2 + h2
	return (xs1 < xs2) and (ys1 < ys2) and (xe1 > xe2) and (ye1 > ye2)

# Read image, convert to black/white, find the contours, remove  outline of image
img = cv2.imread('example_01.jpg')
new_img =img.copy() 
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,127,255,0)
img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda contour: get_roi_area_coords(contour)[1], reverse=True)
# cv2.drawContours(img, contours, 0, (0,255,0), 3)
# cv2.imshow('img', img)
# cv2.waitKey(5000)
contours = contours[1:]


# Find out_contours and in_contours (list of indices)
levels = defaultdict(list)
for i in range(len(contours)-1):
	for j in range(i, len(contours)):
		contour1, contour2 = contours[i], contours[j]
		if inside(contour1,contour2):
			levels[i].append(j)
		elif inside(contour2,contour1):
			levels[j].append(i)

in_contours = list(levels.values())
in_contours = [_[0] for _ in in_contours]
out_contours = [_ for _ in range(len(contours)) if _ not in in_contours]


# Show all out contours
# for i in out_contours:
# 	contour = contours[i]
# 	roi, area, coords = get_roi_area_coords(contour)
# 	xs, ys, xn, yn = coords
# 	rect = cv2.rectangle(img, (xs, ys), (xn, yn), (0, 255, 0), 2)
# 	cv2.imshow('rect', rect)

# cv2.waitKey(5000)

# Cover selected characters in white
coords = [0, 0, 500, 500]
x3,y3 = [50, 50]
#coords = [x1, y1, x2, y2]
selected_contours = []
for i in out_contours:
	contour = contours[i]
	if inside_select(coords,contour):
		selected_contours.append(i)

#print(selected_contours)
for i in selected_contours:
	contour = contours[i]
	roi, area, coords = get_roi_area_coords(contour)
	xs, ys, xn, yn = coords
	cv2.rectangle(new_img,(xs, ys),(xn, yn),(255,255,255),-1)
	#r = cv2.rectangle(img,(xs, ys),(xn, yn),(0,255,0),-1)
	#cv2.imshow('ddd', r)

#cv2.waitKey(1000)

x1 = y1 = 0
dx, dy = x3-x1, y3-y1
# Draw the new contours
for i in selected_contours:
	print(i)
	contour = contours[i]
	roi, area, coords = get_roi_area_coords(contour)
	xs, ys, xe, ye = coords
	print(coords)
	#print(new_img.shape)
	#print(img.shape)
	#print(new_img[xs+dx:xe+dx, ys+dy:ye+dy].shape)
	#print(img[xs:xe, ys:ye].shape)
	new_img[ys+dy:ye+dy,xs+dx:xe+dx] = img[ys:ye, xs:xe]
	print(xs+dx,xe+dx, ys+dy,ye+dy)
	print(xs,xe, ys,ye)
	cv2.imwrite('new' + str(i) + '.jpg',new_img)

# Save the image
new_img[0:50,0:50] = img[50:100,50:100]
cv2.imwrite('new.jpg',new_img)
print('Done')








