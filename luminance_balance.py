import numpy as np
import cv2

'''
Credit: 
This idea is based on the paper
An Image Segmentation Thresholding Method Based on Luminance Proportion
https://wenku.baidu.com/view/f74cc087e53a580216fcfe52.html?from=search
'''


'''This will reduce the luminance range'''
def luminance_local_based(gray):
    local = gray.reshape((48,64,10,10))
    local_mean = np.mean(local, axis = (2,3))
    local_residue = (local.reshape((48 * 64, -1)).T - local_mean.reshape(48 * 64)).T
    local_residue.shape = (480,640)
    global_mean = np.mean(gray)
    output = (local_residue + global_mean).astype(np.uint8)
    return output

'''This will enlarge the luminance range'''
def luminance_global_based(gray):
    local = gray.reshape((48,64,10,10))
    local_mean = np.mean(local, axis = (2,3))
    global_mean = np.mean(gray)
    local_residue = local - global_mean
    output = (local_residue.reshape((48*64, -1)).T + local_mean.reshape(48*64)).T
    output = output.reshape((480, 640))
    output = output.astype(np.uint8)
    return output

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = luminance_local_based(gray)
    output = cv2.GaussianBlur(output, (11,11), 1)
    cv2.imshow('local', output)
    cv2.imshow('origin', gray)
    cap.release()


