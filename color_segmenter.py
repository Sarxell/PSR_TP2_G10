from functools import partial
import numpy as np
import json
import cv2

def onTrackBar(value, channel, min_max, ranges):
    ranges[channel][min_max] = value

def main():
    
    capture = cv2.VideoCapture(0)
    window_name = 'original'
    segmented_window = 'Color Segmenter'
    _, frame = capture.read()    

    ranges={'B':{'min':0, 'max':255},
            'G':{'min':0, 'max':255},
            'R':{'min':0, 'max':255}
    }

    cv2.imshow(segmented_window, frame)

    #each trackbar is created with a callback to onTrackBar with set values for the channel and min_max variables, that will be used to update the corresponding position in the limits dictionary 
    cv2.createTrackbar('MinB', segmented_window, 0, 255, partial(onTrackBar, channel='B', min_max='min', ranges=ranges))
    cv2.createTrackbar('MaxB', segmented_window, 0, 255, partial(onTrackBar, channel='B', min_max='max', ranges=ranges))
    cv2.createTrackbar('MinG', segmented_window, 0, 255, partial(onTrackBar, channel='G', min_max='min', ranges=ranges))
    cv2.createTrackbar('MaxG', segmented_window, 0, 255, partial(onTrackBar, channel='G', min_max='max', ranges=ranges))
    cv2.createTrackbar('MinR', segmented_window, 0, 255, partial(onTrackBar, channel='R', min_max='min', ranges=ranges))
    cv2.createTrackbar('MaxR', segmented_window, 0, 255, partial(onTrackBar, channel='R', min_max='max', ranges=ranges))

    #set position for max TrackBars
    cv2.setTrackbarPos('MaxB', segmented_window, 255)
    cv2.setTrackbarPos('MaxG', segmented_window, 255)
    cv2.setTrackbarPos('MaxR', segmented_window, 255)

    while True:
        _, frame = capture.read()
        # HSV convert
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mins = np.array([ranges['B']['min'], ranges['G']['min'], ranges['R']['min']])
        maxs = np.array([ranges['B']['max'], ranges['G']['max'], ranges['R']['max']])

        mask = cv2.inRange(hsv, mins, maxs)

        cv2.imshow(segmented_window, mask)

       #reading keys
        key=cv2.waitKey(20)
        if key != -1:
            if key == ord('w'):
                # writes in the file limits.json
                file_name = 'limits.json'
                with open(file_name, 'w') as file_handle:
                    print('writing dictionary d to file ' + file_name)
                    json.dump(ranges, file_handle)  # d is the dicionary

            if key == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
