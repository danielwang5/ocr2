import numpy as np
import cv2
from collections import defaultdict
<<<<<<< HEAD
from itertools import chain

show_identified_characters = False
remove = True
add = True
write_step = True
image = 'example_04.jpg'


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
	xs2, ys2, w2, h2 = cv2.boundingRect(contour)
	xe2, ye2 = xs2 + w2, ys2 + h2
	return (xs1 < xs2) and (ys1 < ys2) and (xe1 > xe2) and (ye1 > ye2)


def find_inside_region(contours,i):
	mask = np.zeros_like(img)
	cv2.drawContours(mask, contours, i, (1,1,1), -1)
	return mask

def find_inter_region(contours,i):
	mask = find_inside_region(contours,i)
	if levels[i] == []:
		return mask
	else:
		for child in levels[i]:
			child_mask = find_inside_region(contours,child)
			mask -= child_mask
	#print(mask)
	return mask


def find_nearest_color_2(img,mask,coord):
	x,y = coord
	while np.all(mask[x][y]):
		y += 1
	if not np.all(mask[x][y]) and not np.all(mask[x][y+1]) and not np.all(mask[x][y+2]):
		return img[x,y+1]
	return (255,255,255)




# Read image, convert to black/white, find the contours, remove  outline of image
img = cv2.imread(image)
new_img =img.copy() 
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,127,255,0)
img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda contour: get_roi_area_coords(contour)[1], reverse=True)
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
in_contours = list(chain.from_iterable(in_contours))
out_contours = [_ for _ in range(len(contours)) if _ not in in_contours]
=======
import click
import random

@click.command()
@click.option('--image', type=str, default='example_01.jpg', help='Image to be modified')
@click.option('--remove/--no-remove', default=True, help='Remove the text')
@click.option('--add/--no-add', default=True, help='Add the text back')
@click.option('--write-step/--no-write-step', default=True, help='Include the write step')
@click.option('--show-identified-characters/--no-show-identified-characters', default=False)

#global vars
#num_clicks = 0
#click_list = []

def main(image, remove, add, write_step, show_identified_characters):

	#show_identified_characters = False
	#remove = True
	#add = True
	#write_step = True
	#image = 'example_03.jpg'


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
		xs2, ys2, w2, h2 = cv2.boundingRect(contour)
		xe2, ye2 = xs2 + w2, ys2 + h2
		return (xs1 < xs2) and (ys1 < ys2) and (xe1 > xe2) and (ye1 > ye2)


	def find_inside_region(contours,i):
		mask = np.zeros_like(img)
		cv2.drawContours(mask, contours, i, (1,1,1), -1)
		return mask
>>>>>>> 3c31c529b260f73c4cc49052a79373644c342734

	def find_inter_region(contours,i):
		mask = find_inside_region(contours,i)
		if levels[i] == []:
			return mask
		else:
			for child in levels[i]:
				child_mask = find_inside_region(contours,child)
				mask -= child_mask
		#print(mask)
		return mask

	def find_nearest_color(img,mask,coord):

		buff = random.randint(2,5)

		color_list = []

		x,y = coord
		distR = 0
		while np.all(mask[x][y]):
			distR += 1
			y += 1
		color_list.append((x,y+buff,distR))

		x,y = coord
		distL = 0
		while np.all(mask[x][y]):
			distL += 1
			y -= 1
		color_list.append((x,y-buff,distL))

		x,y = coord
		distU = 0
		while np.all(mask[x][y]):
			distU += 1
			x -= 1
		color_list.append((x-buff,y,distU))

		x,y = coord
		distD = 0
		while np.all(mask[x][y]):
			distD += 1
			x += 1
		color_list.append((x+buff,y,distD))

		total = 1/(distL*distL) + 1/(distR*distR) + 1/(distU*distU) + 1/(distD*distD)

		rand = random.random() * total

		if np.all(mask[color_list[0][0]][color_list[0][1]]) or sum(img[color_list[0][0],color_list[0][1]]) < 30:
			rand *= (rand - 1/(distR*distR))/rand
		else:
			if rand <= 1/(distR*distR):
				return img[color_list[0][0],color_list[0][1]]
			rand -= 1/(distR*distR)

		if np.all(mask[color_list[1][0]][color_list[1][1]]) or sum(img[color_list[1][0],color_list[1][1]]) < 30:
			rand *= (rand - 1/(distL*distL))/rand
		else:
			if rand <= 1/(distL*distL):
				return img[color_list[1][0],color_list[1][1]]
			rand -= 1/(distL*distL)

		if np.all(mask[color_list[2][0]][color_list[2][1]]) or sum(img[color_list[2][0],color_list[2][1]]) < 30:
			rand *= (rand - 1/(distU*distU))/rand
		else:
			if rand <= 1/(distU*distU):
				return img[color_list[2][0],color_list[2][1]]
			rand -= 1/(distU*distU)

		if np.all(mask[color_list[3][0]][color_list[3][1]])  or sum(img[color_list[3][0],color_list[3][1]]) < 30:
			rand *= (rand - 1/(distD*distD))/rand
		else:
			if rand <= 1/(distD*distD):
				return img[color_list[3][0],color_list[3][1]]
			rand -= 1/(distD*distD)

		# if not np.all(mask[x][y]) and not np.all(mask[x][y+1]) and not np.all(mask[x][y+2]):
		# 	return img[x,y+1]
		return (255,255,255)




	# Read image, convert to black/white, find the contours, remove  outline of image
	img = cv2.imread(image)
	new_img =img.copy() 
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(img_gray,127,255,0)
	img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda contour: get_roi_area_coords(contour)[1], reverse=True)
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

	#num_clicks = 0
	#click_list = []

	# Click code
	def on_click(event,x,y,flags,param):
		global num_clicks, click_list

		if event == cv2.EVENT_LBUTTONDBLCLK:
			#cv2.circle(img,(x,y),100,(255,0,0),-1)
			print('x = %d, y = %d'%(x, y))

			num_clicks += 1
			click_list = click_list + [x,y]

			if num_clicks == 3:
				cv2.setMouseCallback('image', lambda *args : None)
				#cv2.destroyAllWindows()

	cv2.namedWindow('image')
	cv2.setMouseCallback('image',on_click)
	img = cv2.imread(image)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	coords = click_list[:4]
	x3,y3 = click_list[4:]
	x1,y1,x2,y2 = coords
	print(coords)
	dx, dy = x3-x1, y3-y1

	# Show all out contours
	if show_identified_characters:
		for i in out_contours:
			contour = contours[i]
			roi, area, coords = get_roi_area_coords(contour)
			xs, ys, xn, yn = coords
			rect = cv2.rectangle(img, (xs, ys), (xn, yn), (0, 255, 0), 2)
			cv2.imshow('rect', rect)

		cv2.waitKey(5000)


	selected_contours = []
	for i in out_contours:
		contour = contours[i]
<<<<<<< HEAD
		roi, area, coords = get_roi_area_coords(contour)
		xs, ys, xn, yn = coords
		rect = cv2.rectangle(img, (xs, ys), (xn, yn), (0, 0, 255), 2)
		cv2.imshow('rect', rect)
		cv2.imwrite(image[:-4] + '_boxes.jpg', rect)
	cv2.waitKey(5000)


selected_contours = []
for i in out_contours:
	contour = contours[i]
	if inside_select(coords,contour):
		selected_contours.append(i)
print(selected_contours)


# Remove selected contours
if remove:
	for i in selected_contours:
		contour = contours[i]
		mask = find_inter_region(contours,i)
		for x in range(len(mask)):
			for y in range(len(mask[0])):
				if np.all(mask[x][y]):
					#print(find_nearest_color(img,mask,[x,y]))
					new_img[x,y] = find_nearest_color(img,mask,[x,y])
		#cv2.imshow('aaa',new_img)	
		#cv2.waitKey(0)

# Draw the new contours
if add:
	n = 0
	for i in selected_contours:
		n+= 1
		print(i)
		contour = contours[i]
		mask = find_inter_region(contours,i)
		for x in range(len(mask)):
			for y in range(len(mask[0])):
				if np.all(mask[x][y]):
					new_img[x+dx,y+dy] = img[x,y]
		if write_step:
			cv2.imwrite(image[:-4] + '_' + str(n) + '.jpg', new_img)

# Save the image
cv2.imwrite(image[:-4] + '_new.jpg', new_img)
print('Done')


=======
		if inside_select(coords,contour):
			selected_contours.append(i)
	print(selected_contours)


	# Remove selected contours
	if remove:
		for i in selected_contours:
			contour = contours[i]
			mask = find_inter_region(contours,i)
			for x in range(len(mask)):
				for y in range(len(mask[0])):
					if np.all(mask[x][y]):
						#print(find_nearest_color(img,mask,[x,y]))
						new_img[x,y] = find_nearest_color(img,mask,[x,y])
						img[x,y] = new_img[x,y]
						mask[x,y] = 0
			#cv2.imshow('aaa',new_img)	
			#cv2.waitKey(0)

	# Draw the new contours
	if add:
		n = 0
		for i in selected_contours:
			n+= 1
			print(i)
			contour = contours[i]
			mask = find_inter_region(contours,i)
			for x in range(len(mask)):
				for y in range(len(mask[0])):
					if np.all(mask[x][y]):
						new_img[x+dx,y+dy] = img[x,y]
			if write_step:
				cv2.imwrite(image[:-4] + '_' + str(n) + '.jpg', new_img)

	# Save the image
	cv2.imwrite(image[:-4] + '_new.jpg', new_img)
	print('Done')
>>>>>>> 3c31c529b260f73c4cc49052a79373644c342734

num_clicks = 0
click_list = []

if __name__ == '__main__':
	main()


