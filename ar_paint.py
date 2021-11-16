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
        
    #this event is purely for testing centroid coords
    if event == cv2.EVENT_RBUTTONDOWN:
        print("coords")
        print(x,y)
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

def centroid(edges, frame):
    #contours holds all objects identified, each object is composed of its edges coordinates
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print ('Number of contours found = ', len(contours))

        #handles the case of not detecting any objects
        if contours!=[]:
            #if it finds objects, it will sort them from biggest to smallest
            sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
            #the biggest will be the first on the list
            biggest_object = sorted_contours[0]

            #list of all x coordinates for biggest_object edges
            center_x_raw=[coord[0][0] for coord in biggest_object]
            #list of all y coordinates for biggest_object edges
            center_y_raw=[coord[0][1] for coord in biggest_object]

            #avg x value, used as centroid x coord
            center_x=int(sum(center_x_raw)/len(center_x_raw))
            #avg y value, used as centroid y coord
            center_y=int(sum(center_y_raw)/len(center_y_raw))

            #drawing the marker for the centroid
            cv2.circle(frame, (center_x, center_y), 10, (255,0,0), 5)

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

    #setup da camera
    vs = VideoStream(0).start()
    frame = vs.read()
    #size of the image from camera
    (h, w) = frame.shape[:2]

    # white canvas
    img = np.zeros((h, w, 3), np.uint8)
    img.fill(255)
    cv2.imshow('canvas', img)
    #mask, gray e video só estou aqui para conseguir testar os mousecallbacks nessas janelas, são para ser removidos depois
    cv2.imshow('gray', img)
    cv2.imshow('mask', img)
    cv2.imshow('video', img)

    # starts red
    color = (0, 0, 255)
    # THICCCCCC
    thickness = 1

     # drawing in the canvas
    cv2.setMouseCallback('canvas', partial(line_drawing, img, color, thickness))
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

        #trying to use this to define edges of recognized objects and track only the biggest one
        edges= cv2.Canny(mask, 200, 180)

        #calculating centroid and placing marker
        centroid(edges, frame)
        #show video, canvas, mask
        cv2.imshow('video', frame)
        cv2.imshow('canvas', img)
        cv2.imshow('mask', mask)
        cv2.imshow('gray', edges)


        key = cv2.waitKey(1)
        if key != -1:
            # choices for changing color
            if key == ord('r'):
                color = (0, 0, 255)
            if key == ord('b'):
                color = (255, 0, 0)
            if key  == ord('g'):
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

            #quit the program
            if key == ord('q'):
                break  
              

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()