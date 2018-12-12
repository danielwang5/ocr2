import numpy as np
import cv2
import signal, os

num_clicks = 0
click_list = []

def handler(signum, frame):
    print ('Signal handler called with signal', signum)

# Set the signal handler
signal.signal(signal.SIGINT, handler)

# python Documents\6.819\ocr2\img_click_test.py

def main():

	def on_click(event,x,y,flags,param):
		global num_clicks, click_list

		if event == cv2.EVENT_LBUTTONDBLCLK:
			cv2.circle(img,(x,y),100,(255,0,0),-1)
			print('x = %d, y = %d'%(x, y))

			num_clicks += 1
			click_list = click_list + [x,y]

			if num_clicks == 3:
				cv2.setMouseCallback('image', lambda *args : None)
				cv2.destroyAllWindows()

	cv2.namedWindow('image')
	cv2.setMouseCallback('image',on_click)

	img = cv2.imread('example_01.jpg')
	cv2.imshow('image',img)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()



if __name__ == '__main__':
	main()