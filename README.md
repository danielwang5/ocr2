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
6. Final image is save in the directory

### 1. Detecting Contours
We used OvenCV2's contour detection API function in order to detect dark text on a light background. Our output is a pixel by pixel list of the boundary of each letter.

### 2. Finding Children Contours
For each contour we find, we may sort by x-coordinate and do a simple line sweep to find the first parent contour. From this, we examine the bounding rectangle of the contour by pooling the min and max x-y coordinates of the contour points, and examing if other contour's bounding rectangles are fully within the parent. At the end of this process, we have a list of parent contours and a list of children contours (we will see later that which child belongs to which parent is irrelevent).

### 3. Grabbing the Pixels in the Letters
We iterate through the rows. For each 1xn pixel strip, we examine the parent contours present on that strip. If a contour is found, the boolean flag switches from "out" to "in", or vice versa. Notice that since the countours don't intersect, we may produce a row of 1s and 0s, where 1 represents "in" and 0 represents "out". We do this for every row to create a binary array A that determines which pixels are inside a parent contour.

We do a similar process for the child contours to create a binary array B for pixels inside the inner contours. Since B is a subset of A, in order to get the pixels in the final letters, we do **C = A XOR B** to get our result.

### 4. Moving Contours
Our user inputs is a bounding box (which encapsulates the contours to be moved) and an additional point P that represents the top left corner of the moved bounding box. This is pretty straightforward, because we simply translate and overwrite each pixel based on C. The more challenging part is cleverly filling the missing pixels in the original location as a result from moving the contours.

### 5. Filling Missing Background Information
We use a simple nearest neighbor process to fill in the missing pixels. Each step, we get the border pixels of the missing segment and assign it the same color as its directly adjacent neighbor. We continue doing this until all pixels are filled. Note that this process is more effective when the background has unsharp gradients.
  
## Additional Features

* Use click Python library to transform the program into a Command Line Interface
* Implemented OpenCV2's mouse event handlers for clicking images to get bounding box and new position inputs
* Takes care of out-of-bound edge cases
