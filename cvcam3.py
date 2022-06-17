#!/usr/bin/python3

import cv2
import time
import imutils
import numpy as np
import configparser

class CvCam(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None

    def run(self):
        ret, self.frame = self.cap.read()
        self.frame = imutils.resize(self.frame, 160)
        return self.frame

    def shutdown(self):
        time.sleep(1)
        self.cap.release()



class ImgHSV():
    def __init__(self):
        # green
        self.gl = np.array([72,  0,   213])
        self.gu = np.array([110, 149, 255])

        # red
        self.rl = np.array([170, 200, 200])
        self.ru = np.array([179, 255, 255])

        self.color = None


    def run(self, img_arr):
        try:
            hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
            g_mask = cv2.inRange(hsv, self.gl, self.gu)
            r_mask = cv2.inRange(hsv, self.rl, self.ru)

            (binary, g_contours, hierarchy) = cv2.findContours(g_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            (binary, r_contours, hierarchy) = cv2.findContours(r_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(g_contours) > len(r_contours):
                contours = g_contours
                self.color = "green"
            elif len(g_contours) < len(r_contours):
                contours = r_contours
                self.color = "red"
            else:
                contours = []
                self.color = None


            if len(contours) > 0:
                try:
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(img_arr, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                    cv2.circle(img_arr, center, 5, (0, 0, 255), -1) 

                except Exception as e:
                    print(e)
                    pass

        except Exception as e:
            print(e)
            pass

        return img_arr





