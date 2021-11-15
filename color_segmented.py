#!/usr/bin/python3

import argparse
import cv2
import numpy as np
from functools import partial
import readchar
import json
from imutils.video import VideoStream


def onTrackbar(val):
    pass

def main():
    vs = VideoStream(0).start()
    window_name = 'original'
    segmented_window = 'segmented'
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, frame)

    ranges = {'limits': {'B': {'max': 255, 'min': 0},
                        'G': {'max': 255, 'min': 0},
                        'R': {'max': 255, 'min': 229}}}

    mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
    maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])

    cv2.namedWindow(segmented_window)

    cv2.createTrackbar('min B/H', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max B/H', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('min G/S', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max G/S', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('min R/V', segmented_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max R/V', segmented_window, 229, 255, onTrackbar)

    while True:
        frame = vs.read()
        cv2.imshow(window_name, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # getting the values of the trackbars
        mins[0] = cv2.getTrackbarPos('min B/H', segmented_window)
        mins[1] = cv2.getTrackbarPos('min G/S', segmented_window)
        mins[2] = cv2.getTrackbarPos('min R/V', segmented_window)
        maxs[0] = cv2.getTrackbarPos('max B/H', segmented_window)
        maxs[1] = cv2.getTrackbarPos('max G/S', segmented_window)
        maxs[2] = cv2.getTrackbarPos('max R/V', segmented_window)

        [ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']] = mins
        [ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']] = maxs

        mask = cv2.inRange(frame, mins, maxs)

        cv2.imshow(segmented_window, mask)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print('writing dictionary d to file ' + file_name)
                json.dump(str(ranges), file_handle)  # d is the dicionary

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    main()