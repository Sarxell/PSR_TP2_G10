import argparse
import cv2
import json
import numpy as np
from functools import partial
from imutils.video import VideoStream

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


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

    return img2, x ,y


# mouse callback function
def line_drawing(img, color, thickness, event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    # user presses the left button
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    # this event is purely for testing centroid coords
    if event == cv2.EVENT_RBUTTONDOWN:
        print("coords")
        print(x, y)
        exit()

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
    cv2.imshow('canvas', img)

    # mask, gray e video só estou aqui para conseguir testar os mousecallbacks nessas janelas, são para ser removidos depois
    # cv2.imshow('gray', img)
    cv2.imshow('mask', img)
    cv2.imshow('video', img)

    # starts red
    color = (0, 0, 255)

    # THICCCCCC
    thickness = 1

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
        print((x, y))

        # drawing the marker for the centroid
        if x:
            cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 5)



        # show video, canvas, mask
        cv2.imshow('video', frame)
        cv2.imshow('canvas', img)
        cv2.imshow('mask', mask)
        cv2.imshow('mask_biggest object', mask_size)

        key = cv2.waitKey(1)

        # drawing in the canvas
        #it is needed in the while for it to change color and thickness, or that or using global variables
        cv2.setMouseCallback('canvas', partial(line_drawing, img, color, thickness))

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

        # change the thickness
        if key == ord('+'):
            thickness += 1
        if key == ord('-'):
            thickness -= 1
            if thickness == 0:
                print('thickness cant be zero')
                thickness = 1

        # quit the program
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()