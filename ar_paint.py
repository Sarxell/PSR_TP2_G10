import argparse
import cv2
import json
import ast
import numpy as np
from functools import partial
from imutils.video import VideoStream

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


# mouse callback function
def line_drawing(img, color, thickness, event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    # user presses the left button
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    # starts drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
            pt1_x, pt1_y = x, y
    # stops drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)


def main():

    parser = argparse.ArgumentParser(description='OPenCV example')
    parser.add_argument('-j', '--json', required=True, type=str, help='Full path to json file')
    args = vars(parser.parse_args())

    # leitura do ficheiro json
    ranges = json.load(open(args['json']))

    print(ranges)
    # min and max values in the json file
    mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
    maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])

    #setup da camera
    vs = VideoStream(0).start()
    frame = vs.read()
    #size of the image from camera
    (h, w) = frame.shape[:2]

    # white canvas
    img = np.zeros((h, w, 3), np.uint8)
    img.fill(255)
    cv2.imshow('canvas', img)
    # starts red
    color = (0, 0, 255)
    # THICCCCCC
    thickness = 1

    while True:
        frame = vs.read()
        # converts frames to HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # creates the mask with the values
        mask = cv2.inRange(gray, mins, maxs)

        #show video, canvas, mask
        cv2.imshow('video', frame)
        cv2.imshow('canvas', img)
        cv2.imshow('mask', mask)

        key = cv2.waitKey(1)
        # choices for changing color
        if key & 0xFF == ord('r'):
            color = (0, 0, 255)
        if key & 0xFF == ord('b'):
            color = (255, 0, 0)
        if key & 0xFF == ord('g'):
            color = (0, 255, 0)

        # clear the board
        if key & 0xFF == ord('c'):
            img.fill(255)

        # change the thickness
        if key & 0xFF == ord('+'):
            thickness += 1
        if key & 0xFF == ord('-'):
            thickness -= 1
            if thickness == 0:
                print('thickness cant be zero')
                thickness = 1

        # drawing in the canvas
        cv2.setMouseCallback('canvas', partial(line_drawing, img, color, thickness))

        #quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()











if __name__ == '__main__':
    main()