# ShapeTracker

## Intro
This is an attempt to track a special pattern with traditional CV methods. It didn't achieve an satisfying outcomes. Thus we went to worship the almighty machine learning.

![The special pattern](https://github.com/AzulRadio/OpenCV-ShapeTracker/blob/master/Resource/armor_board.png)

Armor board for Robomaster, the pattern we are tracking.
<br>
>Not good... at all.

gray:

<img width="400" src="https://github.com/AzulRadio/OpenCV-ShapeTracker/blob/master/Resource/demo_gray.png"/>

output:

<img width="400" src="https://github.com/AzulRadio/OpenCV-ShapeTracker/blob/master/Resource/demo_output.png"/>


<br>

Still, I think it's worth the effort to record what we have done.

## Description
- tracking.py
This is the main script. 
- noise_generator.py
Add noise to the raw image
Someone wants me to test how the model works under noisy conditions.
- luminance_balance.py
Tranditional CV methods are highly sensitive to luminance. So I read a paper about luminance balance and implemented one of them.

## Processing steps
1. read image from camera.
2. RGB to grayscale.
3. Add noise, guassian and sp available.
4. Add a binary threshold.
5. cv2.findContours()
6. rule out contours with area smaller than 50 or greater than 100000
7. rule out contours with more than 16 corners
8. find the min rectangle for contours left, box them and draw on the image.
9. output.
