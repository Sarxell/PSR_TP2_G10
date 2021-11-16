#!/usr/bin/python3.8

import json
import cv2
import numpy as np


def onTrackbar(x):
    pass


def main():
    window_name = 'original'
    cv2.namedWindow(window_name)
    mask_window = 'mask'
    cv2.namedWindow(mask_window)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 15)

    ranges = {'limits': {'B': {'max': 255, 'min': 0},
                         'G': {'max': 255, 'min': 0},
                         'R': {'max': 255, 'min': 0}}}

    mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
    maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])

    cv2.createTrackbar('min B/H', mask_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max B/H', mask_window, 255, 255, onTrackbar)
    cv2.createTrackbar('min G/S', mask_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max G/S', mask_window, 255, 255, onTrackbar)
    cv2.createTrackbar('min R/V', mask_window, 0, 255, onTrackbar)
    cv2.createTrackbar('max R/V', mask_window, 255, 255, onTrackbar)

    while True:
        _, image = capture.read()
        cv2.imshow(window_name, image)

        ranges['limits']['B']['min'] = mins[0] = cv2.getTrackbarPos('min B/H', mask_window)
        ranges['limits']['G']['min'] = mins[1] = cv2.getTrackbarPos('min G/S', mask_window)
        ranges['limits']['R']['min'] = mins[2] = cv2.getTrackbarPos('min R/V', mask_window)
        ranges['limits']['B']['max'] = maxs[0] = cv2.getTrackbarPos('max B/H', mask_window)
        ranges['limits']['G']['max'] = maxs[1] = cv2.getTrackbarPos('max G/S', mask_window)
        ranges['limits']['R']['max'] = maxs[2] = cv2.getTrackbarPos('max R/V', mask_window)

        mask_black = cv2.inRange(image, mins, maxs)
        cv2.imshow(mask_window, mask_black)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print('writing dictionary ranges to file ' + file_name)
                json.dump(ranges, file_handle)  # d is the dictionary
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()