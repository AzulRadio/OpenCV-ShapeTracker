import cv2
import numpy as np

def sp_noise(image,prob):
    thres = 1 - prob
    noise = np.random.rand(*image.shape)
    output = image.copy()
    output[noise < prob] = 0
    output[noise > thres] = 255
    return output

def guassian_noise(image, mean=0, std=0.25):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, std, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return out


if __name__ == "__main__":
    def nothing(var):
        pass
    cap = cv2.VideoCapture(0)
    prob, std = 100, 100
    
    output = np.zeros((480,640))
    cv2.imshow('product', output)
    cv2.createTrackbar('guassian_std','product',25, 50, nothing)
    cv2.createTrackbar('sp_prob','product',25, 100, nothing)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = sp_noise(gray, prob/1000)
        output = guassian_noise(output, std = std/100)
        cv2.imshow('product', output)
        cv2.imshow('origin', gray)
        
        std = cv2.getTrackbarPos('guassian_std','product')
        prob = cv2.getTrackbarPos('sp_prob','product')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
