import argparse
import cv2
import json
import numpy as np
from imutils.video import VideoStream


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

    while True:
        frame = vs.read()
        cv2.imshow('video', frame)
        cv2.imshow('canvas', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break







if __name__ == '__main__':
    main()