#!/usr/bin/env python3

import rospy
import cv2
import numpy as np

class coral_grabcut():
    BLUE = [255,0,0]

    def __init__(self, img_src, rect_coordinates=(0,0,1,1), mask=None, iterCount=1):
        
        self.src = img_src
        if self.src is None:
            exit("ERROR: reading img_src in coral_image init")

        self.rect_coordinates = rect_coordinates
        
        if mask==None:
            self.mask = np.zeros(self.src.shape[:2], dtype = np.uint8)

        self.iterCount = iterCount

        # self.img2=self.src.copy()
        self.output = np.zeros(self.src.shape, np.uint8)

        cv2.namedWindow('input')
        cv2.namedWindow('output')

        
    def grabcut(self):
        cv2.imshow('input', self.img)
        cv2.imshow('output', self.output)

        print("iterCount", self.iterCount)
        for i in range(self.iterCount):
            try:
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)

                cv2.grabCut(self.src, self.mask, self.rect_coordinates, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                # cv2.grabCut(self.src, self.mask, self.rect_coordinates, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
            except:
                import traceback
                traceback.print_exc()

            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv2.bitwise_and(self.src, self.src, mask=mask2)

        print("Returning grabcut output")
        return self.output   