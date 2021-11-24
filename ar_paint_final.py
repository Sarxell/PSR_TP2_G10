import argparse
import json
from functools import partial
import copy
from datetime import *
from imutils.video import VideoStream
import random
from functions import *

#Takes in arguments for the Json used for calibration and for activating shake prevention 
def main():
    video_flag = 0
    parser = argparse.ArgumentParser(description='OPenCV example')
    parser.add_argument('-j', '--json', required=True, type=str, help='Full path to json file')
    parser.add_argument('-sp', '--use_shake_prevention', action='store_true',
                        help='Applies shake detection to the program')

    args = vars(parser.parse_args())

    # ----------------
    # Inicializações
    # ---------------

    # criação da prints para user friendly
    interface()

    # leitura do ficheiro json
    ranges = json.load(open(args['json']))

    # print(ranges)
    # min and max values in the json file
    mins = np.array([ranges['B']['min'], ranges['G']['min'], ranges['R']['min']])
    maxs = np.array([ranges['B']['max'], ranges['G']['max'], ranges['R']['max']])

    # dicionario do path da imagem
    d = {'Ball_painted.jpg': 'Ball.jpg',
         'amongus_painted.jpg': 'amongus.jpg',
         'snail_painted.jpg': 'snail.jpg'}

    # setup da camera
    vs = VideoStream(0).start()
    frame = vs.read()
    # size of the image from camera
    (h, w) = frame.shape[:2]

    # white canvas
    img = np.zeros((h, w, 3), np.uint8)
    img.fill(255)
    window_name = 'canvas'
    cv2.imshow(window_name, img)

    # mask, gray e video só estou aqui para conseguir testar os mousecallbacks nessas janelas, são para ser removidos depois
    # cv2.imshow('gray', img)
    # cv2.imshow('mask', img)
    # cv2.imshow('video', img)

    # starts red
    color = (0, 0, 255)

    # THICCCCCC
    thickness = 1

    # Shape
    shape = Shape.LINE

    # Juntei para evitar os erros nos testes acionados ao premir a tecla q
    img_color = None

    """
    this block is just testing purposes
    cv2.setMouseCallback('gray', partial(line_drawing, img, color, thickness))
    cv2.setMouseCallback('mask', partial(line_drawing, img, color, thickness))
    cv2.setMouseCallback('video', partial(line_drawing, img, color, thickness))
    """

    # ----------------
    # Execucoes
    # ---------------

    while True:
        frame = vs.read()
        frame = cv2.flip(frame, 1)  # the second arguments value of 1 indicates that we want to flip horizontally
        # converts frames to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # creates the mask with the values
        mask = cv2.inRange(hsv, mins, maxs)

        mask_size, x, y = removeSmallComponents(mask, 500)
        # paint the biggest object in the original frame
        frame_copy = copy.copy(frame)
        video = frame.copy()
        frame_copy[(mask_size == 255)] = (0, 255, 0)

        # drawing the marker for the centroid, it is a cross
        if x is not None:
            cv2.line(frame_copy, (int(x) - 10, int(y) + 10), (int(x) + 10, int(y) - 10), (0, 0, 255), 5)
            cv2.line(frame_copy, (int(x) + 10, int(y) + 10), (int(x) - 10, int(y) - 10), (0, 0, 255), 5)

        # drawing in the canvas
        if video_flag:
            mask_drawing_video(window_name, video, img, color, thickness, x, y, shape, args['use_shake_prevention'])
            cv2.setMouseCallback('canvas',
                                 partial(line_drawing_video, w_name=window_name, img=video, mask=img, shape=shape,
                                         color=color, thickness=thickness))
        else:
            mask_drawing(window_name, img, color, thickness, x, y, shape, args['use_shake_prevention'])
            cv2.setMouseCallback('canvas',
                                 partial(line_drawing, w_name=window_name, img=img, shape=shape, color=color,
                                         thickness=thickness))

        # show video, canvas, mask
        # cv2.imshow('video', frame)
        # cv2.imshow('video_changed', frame_copy)
        # videos = cv2.vconcat([frame, frame_copy])
        # cv2.imshow(window_name, img)
        videos = np.concatenate((frame, frame_copy), axis=0)
        cv2.imshow('videos', videos)
        masks = np.concatenate((mask, mask_size), axis=0)
        cv2.imshow('masks', masks)

        # it needs to draw in the img but only show the video
        if video_flag:
            video = frame.copy()
            (h_i, w_i) = img.shape[:2]
            if h_i == h:
                video[(img != 255)] = img[(img != 255)]
                cv2.imshow(window_name, video)

        key = cv2.waitKey(1)

        if args['use_shake_prevention'] is True:
            shake_prevention(x, y, past_x, past_y, color, img)

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
            #img = cv2.resize(img, [w, h])
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

        # select shape for drawing on canvas
        if key == ord('s'):  # square
            print("rectangle")
            shape = Shape.RECTANGLE
        if key == ord('f'):  # circle
            print("circle")
            shape = Shape.CIRCLE
        if key == ord('e'):  # ellipse
            shape = Shape.ELLIPSE
        if key == ord('l'):  #
            print("Line")
            shape = Shape.LINE

        # capture image from videostream and set it on canvas to be drawn
        if key == ord('p'):
            img = frame
            cv2.imshow(window_name, img)

        # get a video in the canvas
        if key == ord('m'):
            video_flag = not video_flag

        if key == ord('t'):
            path_bw = random.choice(list(d.values()))
            img = cv2.imread(path_bw)
            cv2.imshow(window_name, img)
            path_color = list(d.keys())[list(d.values()).index(path_bw)]
            img_color = cv2.imread(path_color)

            # saves the image in the directory of the code
        if key == ord('w'):
            # didnt put weekday
            date_img = datetime.now().strftime("%H:%M:%S_%Y")
            cv2.imwrite('image_' + date_img + '.png', img)

        if key == ord('v'):
            if img_color is not None:
                accuracy(img, img_color)

        # see the user panel again
        if key == ord('h'):
            print(Style.BRIGHT + 'Hey! It is me again' + Style.RESET_ALL)
            interface()

        # quit the program
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()