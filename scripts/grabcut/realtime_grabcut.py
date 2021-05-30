import cv2 as cv
import numpy as np
import sys

""" realtime grabcut with predrawn rectangle, removing need for yolo """ 

class App():
    BLUE = [255,0,0]        # rectangle color

    # setting up flags
    rect = (0,0,1,1)
    drawing = False         # flag for drawing curves
    rectangle = False       # flag for drawing rect
    rect_over = True        # flag to check if rect drawn
    rect_or_mask = 0        # flag for selecting rect or mask mode
    thickness = 3           # brush thickness

    def run(self, frame):

        self.img = frame
        self.img2 = self.img.copy()                                 # a copy of original image
        if self.img is None or self.img2 is None:
            exit("ERROR: img not read")

        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8)  # mask initialized to PR_BG
        self.output = np.zeros(self.img.shape, np.uint8)            # output image to be shown

        self.height, self.width = self.img.shape[:2]
        self.rect = (self.width//4, self.height//8, self.width//2, 3*self.height//4)

        # input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.moveWindow('input', self.img.shape[1]+10,90)

        cv.rectangle(self.img, (self.width//4, self.height//8), (3*self.width//4, 7*self.height//8), [255,0,0], thickness=5)
            
        try:
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)

            cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
        
        except:
            import traceback
            traceback.print_exc()

        mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
        self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)

        cv.imshow('input', self.img)
        cv.imshow('output', self.output)

        print('Done')


if __name__ == '__main__':

    grabcutter = App()

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        exit("ERROR: Failed to open camera")

    while True:
        # print("new iter in outer while loop in main")
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            break

        cv.imshow("Frame", frame)

        grabcutter.run(frame)

        if cv.waitKey(1000) == ord('q'):
            break

    print("Terminating program...")
    cv.destroyAllWindows()
