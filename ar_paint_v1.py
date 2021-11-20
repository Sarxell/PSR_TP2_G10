import argparse
import cv2
import json
import numpy as np
from functools import partial
import copy
from datetime import *
from imutils.video import VideoStream
from enum import Enum

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
copied=False
copied_image=None
flag = 0
past_x, past_y = None, None

#Enum for shapes allowed when drawing on canvas
class Shape(Enum):
    RECTANGLE=1
    CIRCLE=2
    ELLIPSE=3
    LINE=4



def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    x = None
    y = None
    img2 = np.zeros(output.shape, dtype = np.uint8)

    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            # to use the biggest
            x, y = centroids[i+1]
            threshold = sizes[i]
            img2[output == i + 1] = 255

    return img2, x, y


# mouse callback function
def line_drawing(event, x, y, flags, param, w_name, img, shape, color, thickness):
    global pt1_x, pt1_y, drawing, copied, copied_image
    # user presses the left button
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    # starts drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if shape is Shape.RECTANGLE:
                if not copied:
                    copied_image=img.copy()
                    copied=True
                cv2.rectangle(copied_image,(pt1_x, pt1_y), (x,y), color, thickness)
                #pt1_x, pt1_y = x, y
                cv2.imshow(w_name, copied_image)
            if shape is Shape.LINE:
                cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
                pt1_x, pt1_y = x, y
                cv2.imshow(w_name, img)

    # stops drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, copied = False, False
        if shape is Shape.LINE:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.RECTANGLE:
            cv2.rectangle(img,(pt1_x, pt1_y), (x,y), color=color, thickness=thickness)
        cv2.imshow(w_name, img)

# mouse callback function
def mask_drawing(w_name, img, color, thickness, x, y):
    # flag to see if is a new line or not
    global flag
    global past_x, past_y

    if x:
        x = int(x)
        y = int(y)
        # it means there is a new line
        if flag == 1:
            cv2.line(img, (x,y), (x, y), color=color, thickness=thickness)
            past_x = x
            past_y = y
            flag = 0
        else:
            # if flag = 0 it's the same line
            cv2.line(img, (past_x, past_y), (x, y), color=color, thickness=thickness)
            past_x = x
            past_y = y

    else:
        # it starts to be a new line again
        flag = 1

    cv2.imshow(w_name, img)



def main():
    global flag
    parser = argparse.ArgumentParser(description='OPenCV example')
    parser.add_argument('-j', '--json', required=True, type=str, help='Full path to json file')
    args = vars(parser.parse_args())

    # leitura do ficheiro json
    ranges = json.load(open(args['json']))

    print(ranges)
    # min and max values in the json file
    mins = np.array([ranges['B']['min'], ranges['G']['min'], ranges['R']['min']])
    maxs = np.array([ranges['B']['max'], ranges['G']['max'], ranges['R']['max']])

    # setup da camera
    vs = VideoStream(0).start()
    frame = vs.read()
    # size of the image from camera
    (h, w) = frame.shape[:2]

    # white canvas
    img = np.zeros((h, w, 3), np.uint8)
    img.fill(255)
    window_name='canvas'
    cv2.imshow(window_name, img)

    # mask, gray e video só estou aqui para conseguir testar os mousecallbacks nessas janelas, são para ser removidos depois
    # cv2.imshow('gray', img)
    cv2.imshow('mask', img)
    cv2.imshow('video', img)

    # starts red
    color = (0, 0, 255)

    # THICCCCCC
    thickness = 1

    #Shape
    shape=Shape.LINE

    """
    this block is just testing purposes
    cv2.setMouseCallback('gray', partial(line_drawing, img, color, thickness))
    cv2.setMouseCallback('mask', partial(line_drawing, img, color, thickness))
    cv2.setMouseCallback('video', partial(line_drawing, img, color, thickness))
    """

    while True:
        frame = vs.read()
        # converts frames to HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # creates the mask with the values
        mask = cv2.inRange(frame, mins, maxs)

        mask_size, x, y = removeSmallComponents(mask, 500)
        # paint the biggest object in the original frame
        frame_copy = copy.copy(frame)
        frame_copy[(mask_size == 255)] = (0, 255, 0)

        # drawing the marker for the centroid, it is a cross
        if x:
            cv2.line(frame_copy, (int(x)-10, int(y)+10), (int(x)+10, int(y)-10), (0,0,255), 5)
            cv2.line(frame_copy, (int(x) + 10, int(y)+10), (int(x) - 10, int(y) - 10), (0, 0, 255), 5)


        mask_drawing(window_name, img, color, thickness, x, y)

        # show video, canvas, mask
        cv2.imshow('video', frame)
        cv2.imshow('video_changed', frame_copy)
        cv2.imshow('mask', mask)
        cv2.imshow('mask_biggest object', mask_size)

        key = cv2.waitKey(1)

        # drawing in the canvas
        #it is needed in the while for it to change color and thickness, or that or using global variables
        cv2.setMouseCallback('canvas', partial(line_drawing, w_name=window_name, img=img, shape=shape, color=color, thickness=thickness))
        #cv2.setMouseCallback('canvas', partial(shape_drawing, img, shape, color, thickness))

        # it isnt needed
        # if key != -1:

        # choices for changing color
        if key == ord('r'):
            color = (0, 0, 255)
        if key == ord('b'):
            color = (255, 0, 0)
        if key == ord('g'):
            color = (0, 255, 0)

        # clear the board
        if key == ord('c'):
            img.fill(255)
            cv2.imshow(window_name, img)

        # change the thickness
        if key == ord('+'):
            thickness += 1
        if key == ord('-'):
            thickness -= 1
            if thickness == 0:
                print('thickness cant be zero')
                thickness = 1

        #select shape for drawing on canvas
        if key== ord('s'): #square
            print("rectangle")
            shape=Shape.RECTANGLE
        if key== ord('f'):#circle
            shape=Shape.CIRCLE
        if key== ord('e'):#ellipse
            shape=Shape.ELLIPSE
        if key==ord('l'):#
            print("Line")
            shape=Shape.LINE

        #capture image from videostream and set it on canvas to be drawn
        if key == ord('p'):
            img=frame
            cv2.imshow(window_name, img)

        # saves the image in the directory of the code
        if key == ord('w'):
            # didnt put weekday
            date_img = datetime.now().strftime("%H:%M:%S_%Y")
            cv2.imwrite('image_' + date_img + '.png', img)

        # quit the program
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()