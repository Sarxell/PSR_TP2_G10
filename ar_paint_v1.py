import argparse
import cv2
import json
import numpy as np
from functools import partial
import copy
from datetime import *
from colorama import Fore, Back, Style
from imutils.video import VideoStream
from enum import Enum
import random
import math

from numpy.lib.function_base import _place_dispatcher

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
copied = False
copied_image = None
copied_image_2 = None
flag = 0
past_x, past_y = None, None
holding = False
finished = False


# Enum for shapes allowed when drawing on canvas
class Shape(Enum):
    RECTANGLE = 1
    CIRCLE = 2
    ELLIPSE = 3
    LINE = 4


def accuracy(img_bw, img_color):
    ## convert to hsv both our drawing and the painted one
    hsv_bw = cv2.cvtColor(img_bw, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv_bw, (36, 25, 25), (70, 255, 255))
    b = cv2.inRange(hsv_bw, (110, 50, 50), (130, 255, 255))
    r = cv2.inRange(hsv_bw, (0, 50, 70), (9, 255, 255))
    hsv_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    g_color = cv2.inRange(hsv_color, (36, 25, 25), (70, 255, 255))
    b_color = cv2.inRange(hsv_color, (94, 80, 2), (126, 255, 255))
    r_color = cv2.inRange(hsv_color, (159, 50, 70), (180, 255, 255))

    # we also need to remove the small components from the painted mask
    g_color, _, _ = removeSmallComponents(g_color, threshold=20)
    b_color, _, _ = removeSmallComponents(b_color, threshold=20)
    r_color, _, _ = removeSmallComponents(r_color, threshold=20)

    # the masks of every color
    # the part painted that is right
    bitwiseAnd_g = cv2.bitwise_and(g_color, g)
    bitwiseAnd_r = cv2.bitwise_and(r_color, r)
    bitwiseAnd_b = cv2.bitwise_and(b_color, b)
    # ALL THE Paint
    bitwiseOr_g = cv2.bitwise_or(g_color, g)
    bitwiseOr_r = cv2.bitwise_or(r_color, r)
    bitwiseOr_b = cv2.bitwise_or(b_color, b)

    # calculus
    green_painted = sum(sum(bitwiseAnd_g))
    total_green = sum(sum(bitwiseOr_g))
    acc_green = green_painted / total_green * 100

    blue_painted = sum(sum(bitwiseAnd_b))
    total_blue = sum(sum(bitwiseOr_b))
    acc_blue = blue_painted / total_blue * 100

    red_painted = sum(sum(bitwiseAnd_r))
    total_red = sum(sum(bitwiseOr_r))
    acc_red = red_painted / total_red * 100

    total_acc = (blue_painted + green_painted + red_painted) / (total_red + total_blue + total_green) * 100

    print('Your blue accuracy was ' + str(acc_blue))
    print('Your green accuracy was ' + str(acc_green))
    print('Your red accuracy was ' + str(acc_red))
    print('Your total accuracy was ' + str(total_acc))


def removeSmallComponents(image, threshold):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    x = None
    y = None
    img2 = np.zeros(output.shape, dtype=np.uint8)

    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            # to use the biggest
            x, y = centroids[i + 1]
            threshold = sizes[i]
            img2[output == i + 1] = 255

    return img2, x, y


# mouse callback function
def line_drawing(event, x, y, flags, param, w_name, img, shape, color, thickness):
    global pt1_x, pt1_y, drawing, copied, copied_image
    global holding, finished
    # user presses the left button
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    # starts drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # after the rectangle, he disappears if the button wasnt pressed
            copied_image = img.copy()
            if shape is Shape.RECTANGLE:
                if not copied:
                    copied_image = img.copy()
                    copied = True
                cv2.rectangle(copied_image, (pt1_x, pt1_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                if not copied:
                    copied_image = img.copy()
                    copied = True
                cv2.circle(copied_image, (pt1_x, pt1_y),
                           int(math.sqrt(math.pow(x - pt1_x, 2) + math.pow(y - pt1_y, 2))), color, thickness)
            if shape is Shape.LINE:
                cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
                pt1_x, pt1_y = x, y

            if copied:
                cv2.imshow(w_name, copied_image)
            else:
                cv2.imshow(w_name, img)

    # stops drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, copied = False, False
        if shape is Shape.LINE:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.RECTANGLE:
            cv2.rectangle(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.CIRCLE:
            cv2.circle(img, (pt1_x, pt1_y), int(math.sqrt(math.pow(x - pt1_x, 2) + math.pow(y - pt1_y, 2))), color,
                       thickness)

        cv2.imshow(w_name, img)

    if event == cv2.EVENT_MBUTTONDOWN:
        holding = True
        finished = False

    if event == cv2.EVENT_MBUTTONUP:
        holding = False
        finished = True


# mouse callback function
def mask_drawing(w_name, img, color, thickness, x, y, shape):
    # flag to see if is a new line or not
    global flag, holding, finished
    global copied, copied_image
    global past_x, past_y

    if not holding:
        if x:
            x = int(x)
            y = int(y)
            # it means there is a new line
            if flag == 1:
                cv2.line(img, (x, y), (x, y), color=color, thickness=thickness)
                past_x = x
                past_y = y
                flag = 0
            else:
                # if flag = 0 it's the same line
                if not shake_prevention(x,y, past_x, past_y, color, img):
                    if past_x and past_y:
                        cv2.line(img, (past_x, past_y), (x, y), color=color, thickness=thickness)
                        past_x = x
                        past_y = y

        else:
            # it starts to be a new line again
            flag = 1

    else:
        if x:
            x = int(x)
            y = int(y)
            copied_image = img.copy()
            if not finished:
                copied = True
                copied_image = img.copy()
            if shape is Shape.RECTANGLE:
                cv2.rectangle(copied_image , (past_x, past_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                cv2.circle(copied_image , (past_x, past_y),
                           int(math.sqrt(math.pow(x - past_x, 2) + math.pow(y - past_y, 2))), color, thickness)

    if finished:
        finished = False
        copied = False
        if shape is Shape.RECTANGLE:
            cv2.rectangle(img, (past_x, past_y), (x, y), color, thickness)
        if shape is Shape.CIRCLE:
            cv2.circle(img, (past_x, past_y),
                       int(math.sqrt(math.pow(x - past_x, 2) + math.pow(y - past_y, 2))), color, thickness)

    if copied:
        cv2.imshow(w_name, copied_image)
    else:
        cv2.imshow(w_name, img)
   

def shake_prevention(x, y,past_x, past_y, color, img):
    #print('X     ' + str(x))
    #print('PAST     ' + str(past_x))
    #Distancia ponto atual ao ponto anterior
    if past_x and past_y:
        dist=int(math.sqrt(math.pow(x-past_x,2)+math.pow(y-past_y,2)))
        #Se a distancia for superior a 50 retorna que é necessário fazer shake prevention caso contrario retorna que não é necessário
        if dist > 200:
            cv2.circle(img, (x, y), radius = 0, color=color, thickness=-1)
            return True
        return False


def main():
    global flag
    video_flag = 0
    parser = argparse.ArgumentParser(description='OPenCV example')
    parser.add_argument('-j', '--json', required=True, type=str, help='Full path to json file')
    parser.add_argument('-sp', '--use_shake_prevention', action='store_true',
                        help='Applies shake detection to the program')

    args = vars(parser.parse_args())

    # ----------------
    # Inicializações
    # ---------------

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
    cv2.imshow('mask', img)
    cv2.imshow('video', img)

    # starts red
    color = (0, 0, 255)

    # THICCCCCC
    thickness = 1

    # Shape
    shape = Shape.LINE

    # Juntei para evitar os erros nos testes acionados ao premir a tecla q
    # img_color = None

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
        frame_copy[(mask_size == 255)] = (0, 255, 0)

        # drawing the marker for the centroid, it is a cross
        if x is not None:
            cv2.line(frame_copy, (int(x) - 10, int(y) + 10), (int(x) + 10, int(y) - 10), (0, 0, 255), 5)
            cv2.line(frame_copy, (int(x) + 10, int(y) + 10), (int(x) - 10, int(y) - 10), (0, 0, 255), 5)

        # drawing in the canvas
        mask_drawing(window_name, img, color, thickness, x, y, shape)
        cv2.setMouseCallback('canvas',
                             partial(line_drawing, w_name=window_name, img=img, shape=shape, color=color,
                                     thickness=thickness))

        # show video, canvas, mask
        cv2.imshow('video', frame)
        cv2.imshow('video_changed', frame_copy)
        cv2.imshow('mask', mask)
        cv2.imshow('mask_biggest object', mask_size)
        if video_flag:
            video = copy.copy(frame)
            video[(img != 255)] = img[(img != 255)]
            cv2.imshow(window_name, video)

        key = cv2.waitKey(1)

        if args['use_shake_prevention'] is True:
            shake_prevention(x, y,past_x, past_y, color, img)

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
            print(video_flag)

        if key == ord('t'):
            path_bw = random.choice(list(d.values()))
            print(path_bw)
            img = cv2.imread(path_bw)
            cv2.imshow(window_name, img)
            path_color = list(d.keys())[list(d.values()).index(path_bw)]
            img_color = cv2.imread(path_color)
            print(path_color)

            # saves the image in the directory of the code
        if key == ord('w'):
            # didnt put weekday
            date_img = datetime.now().strftime("%H:%M:%S_%Y")
            cv2.imwrite('image_' + date_img + '.png', img)

        if key == ord('v'):
            if img_color is not None:
                accuracy(img, img_color)

        # quit the program
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
