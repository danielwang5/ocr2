# ocr2

## Outline

1. Detect contours
2. Find its children contours (e.g the inner contour of an "o" or a "6")
3. Grab all pixels in parent contour but outside child contour(s)
  * Can do this in a row by row basis with XOR filtering
4. Move text contours to desired location
5. Fill missing background information from moving contours
  * Take neighest neighbor of edge node, and loop until all missing background pixels are filled
  * May also use a neighborhood heuristic instead of a single node
  
  
## Additional Features

* Use click Python library to transform the program into a Command Line Interface
* Implemented OpenCV2's mouse event handlers for clicking images to get bounding box and new position inputs
* Takes care of out-of-bound edge cases
