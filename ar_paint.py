import argparse
import cv2
import json
import numpy as np
from functools import partial
from imutils.video import VideoStream

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


# mouse callback function
def line_drawing(img, color, thickness, event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)


def main():

    parser = argparse.ArgumentParser(description='OPenCV example')
    parser.add_argument('-j', '--json', required=True, type=str, help='Full path to json file')
    args = vars(parser.parse_args())

    # leitura do ficheiro json
    with open(args['json']) as f:
        ranges = json.load(f)
        print(ranges)

    #setup da camera
    vs = VideoStream(0).start()
    frame = vs.read()
    (h, w) = frame.shape[:2]
    img = np.zeros((h, w, 3), np.uint8)
    img.fill(255)
    cv2.imshow('canvas', img)
    # starts red
    color = (0, 0, 255)
    # THICCCCCC
    thickness = 1

    while True:
        frame = vs.read()
        cv2.imshow('video', frame)
        cv2.imshow('canvas', img)

        # choices for changing color, linesize, clear etc...
        key = cv2.waitKey(1)
        if key & 0xFF == ord('r'):
            color = (0, 0, 255)
        if key & 0xFF == ord('b'):
            color = (255, 0, 0)
        if key & 0xFF == ord('g'):
            color = (0, 255, 0)
        if key & 0xFF == ord('c'):
            img.fill(255)
        if key & 0xFF == ord('+'):
            thickness += 1
        if key & 0xFF == ord('-'):
            thickness -= 1
            if thickness == 0:
                print('thickness cant be zero')
                thickness = 1


        cv2.setMouseCallback('canvas', partial(line_drawing, img, color, thickness))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()











if __name__ == '__main__':
    main()