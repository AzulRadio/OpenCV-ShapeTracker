import numpy as np
import cv2
from luminance_balance import luminance_local_based 
def nothing(var):
    pass

cap = cv2.VideoCapture(0)

_, temp = cap.read()

cv2.imshow('frame', np.zeros_like(temp))

cv2.createTrackbar('Threshold','frame',200, 255, nothing)
cv2.createTrackbar('epsilon','frame',50, 100, nothing)

last = np.zeros(2)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    center = np.zeros(2)
    
    # binay and median_blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = luminance_local_based(gray)
    gray = cv2.GaussianBlur(gray, (101,101), 1)
    cv2.imshow('gray',gray)
    thres = cv2.getTrackbarPos('Threshold','frame')
    _, binary = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY_INV)
    binary_blur = cv2.medianBlur(binary, 11)
    output = binary_blur

    #find contours
    contours, hierarchy = cv2.findContours(output,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(0,len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 50 or area > 100000:
            continue
        epsilon_const = cv2.getTrackbarPos('epsilon','frame')
        epsilon = epsilon_const/1000 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        corners = len(approx)
        if corners > 16:
            continue
        #x, y, w, h = cv2.boundingRect(contours[i])
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        output = cv2.drawContours(output,[box],0,(0,255,255),2)
        frame = cv2.drawContours(frame,[box],0,(0,255,255),2)
        
        mm = cv2.moments(contours[i])
        cx = int(mm['m10'] / (mm['m00'] + 1e-6))
        cy = int(mm['m01'] / (mm['m00'] + 1e-6))
        center = np.vstack((center, np.array([cx, cy])))
        cv2.circle(frame, (cx, cy), 3, (100, 255, 100), -1)
    # Display the resulting frame1
    cv2.imshow('frame',frame)
    cv2.imshow('output', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
