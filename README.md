# ocr2

An Optical Character Recognition System (OCR) with Pattern Editing Functionalities

## Goal

Identify characters (or more broadly, patterns) within a scanned image and edit the image by duplicating, removing, or relocating selected characters. 

## Demonstration

Replace a pattern
```
python main2.py --image example_01.jpg
```
Duplicate a pattern
```
python main2.py --image example_01.jpg --no-remove
```
Remove a pattern
```
python main2.py --image example_01.jpg --no-add
```
Show the identified characters
```
python main2.py --image example_01.jpg --show-identified-characters
```
Store a picture for each step during the process
```
python main2.py --image example_01.jpg --write-step
```

## Outline

1. Detect contours in image
2. Establish contour hierarchy (Find children contours) (e.g the inner contour of an "o" or a "6")
   * By generating a dictionary to store the relationship and a list of in_contours and out_contours
3. Retrieve user input to determine characters to be moved and the desired location
4. Find the inter-region of each character
   * Grab all pixels in parent contour but outside child contour(s)
   * Can do this in a row by row basis with XOR filtering
   * Create masks to mark out the inter-region of characters.
5. Remove selected characters and fill in missing background information
   * Remove the pixels in the inter-region of selected characters
   * Take neighest neighbor of edge pixel, and loop until all missing background pixels are filled
   * May also use a neighborhood heuristic method or texture synthesis algorithms instead of a single pixel
6. Redraw characters at the desired location
7. Final image is saved in the working directory

### 1. Detecting Contours
We used OvenCV2's contour detection API function in order to seperate text from the background. The output is a list of contours, each composed of a list of coordinates.

### 2. Establish contour hierarchy (Find children contours)
For each contour we find, we may sort by x-coordinate and do a simple line sweep to find the first parent contour. From this, we examine the bounding rectangle of the contour by pooling the min and max x-y coordinates of the contour points, and examing if other contour's bounding rectangles are fully within the parent. At the end of this process, we have a list of parent contours and a list of children contours (we will see later that which child belongs to which parent is irrelevent).

### 3. Retrieve user input to determine characters to be moved and the desired location

### 4. Find the inter-region of each character
We iterate through the rows. For each 1xn pixel strip, we examine the parent contours present on that strip. If a contour is found, the boolean flag switches from "out" to "in", or vice versa. Notice that since the countours don't intersect, we may produce a row of 1s and 0s, where 1 represents "in" and 0 represents "out". We do this for every row to create a binary array A that determines which pixels are inside a parent contour.

We do a similar process for the child contours to create a binary array B for pixels inside the inner contours. Since B is a subset of A, in order to get the pixels in the final letters, we do **C = A XOR B** to get our result.

### 5. Remove characters and fill in missing background information
We use a simple nearest neighbor process to fill in the missing pixels. Each step, we get the border pixels of the missing segment and assign it the same color as its directly adjacent neighbor. We continue doing this until all pixels are filled. Note that this process is more effective when the background has unsharp gradients.

### 6. Redraw characters at the desired location
Our user inputs is a bounding box (which encapsulates the contours to be moved) and an additional point P that represents the top left corner of the moved bounding box. This is pretty straightforward, because we simply translate and overwrite each pixel based on C. The more challenging part is cleverly filling the missing pixels in the original location as a result from moving the contours.

### 7. Final image is saved in the working directory


## Additional Features

* Use click Python library to transform the program into a Command Line Interface
* Implemented OpenCV2's mouse event handlers for clicking images to get bounding box and new position inputs
* Takes care of out-of-bound edge cases

## Dependencies

1. OpenCV
2. Numpy
3. click
4. collections
5. random


## References
1. Xinyu Zhou et al. (2017). EAST: An Efficient and Accurate Scene Text Detector. arXiv:1704.03155 [cs.CV]
2. Ray Smith (2007). An overview of the Tesseract OCR Engine. Proc. 9th IEEE ICDAR.
3. Gregory Cohen et al. (2017). EMNIST: an extension of MNIST to handwritten letters. arXiv:1702.05373 [cs.CV]
4. Adrian Rosebrock (2018). OpenCV Text Detection (EAST text detector). https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
5. Soumith Chintala. Deep Learning with PyTorch: A 60 Minute Blitz. https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
