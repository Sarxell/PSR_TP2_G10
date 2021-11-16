#!/usr/bin/python3.8

import json
import cv2
import numpy as np

import argparse
import cv2
import numpy as np
import json
# from imutils.video import VideoStream


def onTrackbar(val):
    pass

def main():

    capture = cv2.VideoCapture(0)
    window_name = 'original'
    segmented_window = 'segmented'
    _, frame = capture.read()
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, frame)

    ranges = {'limits': {'B': {'max': 255, 'min': 0},
                         'G': {'max': 255, 'min': 0},
                         'R': {'max': 255, 'min': 229}}}

    # takes the values from the dictionary, could not be needed
    mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
    maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])

    cv2.namedWindow(segmented_window)
    # creates the trackbars
    cv2.createTrackbar('min B/H', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max B/H', segmented_window, 255, 255, onTrackbar)
    cv2.createTrackbar('min G/S', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max G/S', segmented_window, 255, 255, onTrackbar)
    cv2.createTrackbar('min R/V', segmented_window, 229, 255, onTrackbar)
    cv2.createTrackbar('max R/V', segmented_window, 255, 255, onTrackbar)

    while True:
        _, frame = capture.read()
        cv2.imshow(window_name, frame)
        # HSV convert
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # getting the values of the trackbars
        # places the value in the dictionary
        ranges['limits']['B']['min'] = mins[0] = cv2.getTrackbarPos('min B/H', segmented_window)
        ranges['limits']['G']['min'] = mins[1] = cv2.getTrackbarPos('min G/S', segmented_window)
        ranges['limits']['R']['min'] = mins[2] = cv2.getTrackbarPos('min R/V', segmented_window)
        ranges['limits']['B']['max'] = maxs[0] = cv2.getTrackbarPos('max B/H', segmented_window)
        ranges['limits']['G']['max'] = maxs[1] = cv2.getTrackbarPos('max G/S', segmented_window)
        ranges['limits']['R']['max'] = maxs[2] = cv2.getTrackbarPos('max R/V', segmented_window)

        # creates the mask with the values
        mask = cv2.inRange(gray, mins, maxs)
        cv2.imshow(segmented_window, mask)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            # writes in the file limits.json
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print('writing dictionary d to file ' + file_name)
                json.dump(ranges, file_handle)  # d is the dicionary

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    main()