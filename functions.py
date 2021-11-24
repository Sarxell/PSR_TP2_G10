import cv2
import numpy as np
from colorama import Fore, Back, Style
from enum import Enum
import math


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
copied = False
copied_image = None
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


def interface():
    print(Style.BRIGHT + 'Function of every key:')
    print(Style.BRIGHT + 'Colors:')
    print(Style.BRIGHT + Fore.RED + 'Red: ' + Style.RESET_ALL + 'r')
    print(Style.BRIGHT + Fore.BLUE + 'Blue: ' + Style.RESET_ALL + 'b')
    print(Style.BRIGHT + Fore.GREEN + 'Green: ' + Style.RESET_ALL + 'g\n')
    print(Style.BRIGHT + 'Thickness:')
    print(Style.BRIGHT + 'More thickness: ' + Style.RESET_ALL + '+')
    print(Style.BRIGHT + 'Less thickness: ' + Style.RESET_ALL + '-\n')
    print(Style.BRIGHT + 'Drawn with the mask: ' + Style.RESET_ALL + 'middle button hold')
    print(Style.BRIGHT + 'Drawn with the mouse: ' + Style.RESET_ALL + 'left button hold\n')
    print(Style.BRIGHT + 'Shapes:')
    print(Style.BRIGHT + 'Squares: ' + Style.RESET_ALL + 's')
    print(Style.BRIGHT + 'Circles: ' + Style.RESET_ALL + 'f')
    print(Style.BRIGHT + 'Ellipses: ' + Style.RESET_ALL + 'e\n')
    print(Style.BRIGHT + 'Draw in a captured picture: ' + Style.RESET_ALL + 'p')
    print(Style.BRIGHT + 'Draw in the video: ' + Style.RESET_ALL + 'm')
    print(Style.BRIGHT + 'Color paint test: ' + Style.RESET_ALL + 't')
    print(Style.BRIGHT + 'See accuracy of the test: ' + Style.RESET_ALL + 'v\n')
    print(Style.BRIGHT + 'Save the image: ' + Style.RESET_ALL + 'w')
    print(Style.BRIGHT + 'To clear the canvas: ' + Style.RESET_ALL + 'c')
    print(Style.BRIGHT + 'To quit: ' + Style.RESET_ALL + 'q')
    print(Style.BRIGHT + 'To see this panel again use: ' + Style.RESET_ALL + Back.YELLOW + Fore.BLACK + 'h' +
          Style.RESET_ALL)


def distance(current_location, previous_location):
    return int(math.sqrt(
        math.pow(current_location[0] - previous_location[0], 2) + math.pow(current_location[1] - previous_location[1],
                                                                           2)))


def angle(x1, x2, y1, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def accuracy(img_bw, img_color):

    ## convert to hsv both our drawing and the painted one
    hsv_bw = cv2.cvtColor(img_bw, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv_bw, (36, 25, 25), (70, 255, 255))
    b = cv2.inRange(hsv_bw, (110, 50, 50), (130, 255, 255))
    r = cv2.inRange(hsv_bw, (0, 50, 70), (9, 255, 255))
    hsv_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    g_color = cv2.inRange(hsv_color, (36, 25, 25), (70, 255, 255))
    b_color = cv2.inRange(hsv_color, (94, 80, 2), (126, 255, 255))
    r1_color = cv2.inRange(hsv_color, (159, 50, 70), (180, 255, 255))
    r2_color = cv2.inRange(hsv_color, (0, 50, 70), (9, 255, 255))
    r_color = cv2.bitwise_or(r2_color, r1_color)

    # we also need to remove the small components from the painted mask
    kernel = np.ones((5, 5), np.uint8)
    r_color = cv2.erode(r_color, kernel, 1)
    r_color = cv2.dilate(r_color, kernel, 1)
    b_color = cv2.erode(b_color, kernel, 1)
    b_color = cv2.dilate(b_color, kernel, 1)
    g_color = cv2.erode(g_color, kernel, 1)
    g_color = cv2.dilate(g_color, kernel, 1)

    # the masks of every color
    # the part painted that is right
    bitwiseAnd_g = cv2.bitwise_and(g_color, g)
    bitwiseAnd_r = cv2.bitwise_and(r_color, r)
    bitwiseAnd_b = cv2.bitwise_and(b_color, b)
    # ALL THE Paint
    bitwiseor_g = cv2.bitwise_or(g_color, g)
    bitwiseor_r = cv2.bitwise_or(r, r_color)
    bitwiseor_b = cv2.bitwise_or(b, b_color)

    # calculus
    bitwiseor_g[bitwiseor_g > 0] = 1
    bitwiseAnd_g[bitwiseAnd_g > 0] = 1
    green_painted = sum(sum(bitwiseAnd_g))
    total_green = sum(sum(bitwiseor_g))

    acc_green = (green_painted / total_green) * 100

    bitwiseor_b[bitwiseor_b > 0] = 1
    bitwiseAnd_b[bitwiseAnd_b > 0] = 1
    blue_painted = sum(sum(bitwiseAnd_b))
    total_blue = sum(sum(bitwiseor_b))
    acc_blue = (blue_painted / total_blue) * 100

    bitwiseor_r[bitwiseor_r > 0] = 1
    bitwiseAnd_r[bitwiseAnd_r > 0] = 1
    red_painted = sum(sum(bitwiseAnd_r))
    total_red = sum(sum(bitwiseor_r))
    acc_red = (red_painted / total_red) * 100

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
                cv2.circle(copied_image, (pt1_x, pt1_y), distance((x, y), (pt1_x, pt1_y)), color, thickness)
            if shape is Shape.ELLIPSE:
                if not copied:
                    copied_image = img.copy()
                    copied = True
                cv2.ellipse(copied_image, (pt1_x, pt1_y), (abs(x - pt1_x), abs(y - pt1_y)),
                            angle(pt1_x, x, pt1_y, y),
                            0., 360, color, thickness)
            if shape is Shape.LINE:
                cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
                pt1_x, pt1_y = x, y
            else:
                cv2.imshow(w_name, copied_image)

    # stops drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, copied = False, False
        if shape is Shape.LINE:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.RECTANGLE:
            cv2.rectangle(img, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.CIRCLE:
            cv2.circle(img, (pt1_x, pt1_y), distance((x, y), (pt1_x, pt1_y)), color, thickness)
        if shape is Shape.ELLIPSE:
            cv2.ellipse(img, (pt1_x, pt1_y), (abs(x - pt1_x), abs(y - pt1_y)), angle(pt1_x, x, pt1_y, y), 0.,
                        360, color,
                        thickness)

    if event == cv2.EVENT_MBUTTONDOWN:
        holding = True
        finished = False

    if event == cv2.EVENT_MBUTTONUP:
        finished = True


# mouse callback function
def line_drawing_video(event, x, y, flags, param, w_name, img, mask, shape, color, thickness):
    global pt1_x, pt1_y, drawing, copied_image
    global holding, finished
    # user presses the left button
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    # starts drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # after the form, he disappears if the button wasnt pressed
            copied_image = img.copy()
            if shape is Shape.RECTANGLE:
                cv2.rectangle(copied_image, (pt1_x, pt1_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                cv2.circle(copied_image, (pt1_x, pt1_y), distance((x, y), (pt1_x, pt1_y)), color, thickness)
            if shape is Shape.ELLIPSE:
                cv2.ellipse(copied_image, (pt1_x, pt1_y), (abs(x - pt1_x), abs(y - pt1_y)), angle(pt1_x, x, pt1_y, y),
                            0.,360, color, thickness)
            if shape is Shape.LINE:
                cv2.line(mask, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
                pt1_x, pt1_y = x, y
            else:
                cv2.imshow(w_name, copied_image)

    # stops drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if shape is Shape.LINE:
            cv2.line(mask, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.RECTANGLE:
            cv2.rectangle(mask, (pt1_x, pt1_y), (x, y), color=color, thickness=thickness)
        if shape is Shape.CIRCLE:
            cv2.circle(mask, (pt1_x, pt1_y), distance((x, y), (pt1_x, pt1_y)), color, thickness)
        if shape is Shape.ELLIPSE:
            cv2.ellipse(mask, (pt1_x, pt1_y), (abs(x - pt1_x), abs(y - pt1_y)), angle(pt1_x, x, pt1_y, y),
                        0., 360, color, thickness)

    if event == cv2.EVENT_MBUTTONDOWN:
        holding = True
        finished = False

    if event == cv2.EVENT_MBUTTONUP:
        finished = True


# mouse callback function
def mask_drawing(w_name, img, color, thickness, x, y, shape, flag_shake):
    # flag to see if is a new line or not
    global flag, holding, finished
    global copied, copied_image
    global past_x, past_y

    if holding is False:
        if x:
            x = int(x)
            y = int(y)
            if shape is Shape.LINE:
                # it starts a new line
                if flag == 1:
                    cv2.line(img, (x, y), (x, y), color=color, thickness=thickness)
                    past_x = x
                    past_y = y
                    flag = 0
                else:
                    # if flag = 0 it's the same line
                    if flag_shake is True:
                        if not shake_prevention(x, y, past_x, past_y, color, img):
                            if past_x and past_y:
                                cv2.line(img, (past_x, past_y), (x, y), color=color, thickness=thickness)
                                past_x = x
                                past_y = y
                    else:
                        if past_x and past_y:
                            cv2.line(img, (past_x, past_y), (x, y), color=color, thickness=thickness)
                            past_x = x
                            past_y = y
            else:
                past_x = x
                past_y = y
        else:
            flag = 1

    if holding is True:
        if x:
            x = int(x)
            y = int(y)
            if finished is False:
                copied = True
                copied_image = img.copy()
            if shape is Shape.RECTANGLE:
                cv2.rectangle(copied_image, (past_x, past_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                cv2.circle(copied_image, (past_x, past_y), distance((x, y), (past_x, past_y)), color, thickness)
            if shape is Shape.ELLIPSE:
                cv2.ellipse(copied_image, (past_x, past_y), (abs(x + 1 - past_x), abs(y + 1 - past_y)),
                            angle(past_x, x, past_y, y), 0., 360, color, thickness)
            if shape is Shape.LINE:
                cv2.line(img, (past_x, past_y), (x, y), color=color, thickness=thickness)
                past_x = x
                past_y = y
            else:
                cv2.imshow(w_name, copied_image)

    if finished is True:
        finished = False
        copied = False
        holding = False
        if x and past_x:
            if shape is Shape.RECTANGLE:
                cv2.rectangle(img, (past_x, past_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                cv2.circle(img, (past_x, past_y), distance((x, y), (past_x, past_y)), color, thickness)
            if shape is Shape.ELLIPSE:
                cv2.ellipse(img, (past_x, past_y), (abs(x + 1 - past_x), abs(y + 1 - past_y)),
                            angle(past_x, x, past_y, y), 0., 360, color, thickness)

    if copied is True:
        cv2.imshow(w_name, copied_image)
    else:
        cv2.imshow(w_name, img)


# mouse callback function
def mask_drawing_video(w_name, img, mask, color, thickness, x, y, shape, flag_shake):
    # flag to see if is a new line or not
    global flag, holding, finished
    global copied, copied_image
    global past_x, past_y

    if not holding:
        if x:
            x = int(x)
            y = int(y)
            # it means there is a new line
            if shape is Shape.LINE:
                if flag == 1:
                    cv2.line(mask, (x, y), (x, y), color=color, thickness=thickness)
                    past_x = x
                    past_y = y
                    flag = 0
                else:
                    # if flag = 0 it's the same line
                    if flag_shake is True:
                        if not shake_prevention(x, y, past_x, past_y, color, img):
                            if past_x and past_y:
                                cv2.line(mask, (past_x, past_y), (x, y), color=color, thickness=thickness)
                                past_x = x
                                past_y = y
                    else:
                        if past_x and past_y:
                            cv2.line(mask, (past_x, past_y), (x, y), color=color, thickness=thickness)
                            past_x = x
                            past_y = y
            else:
                past_x = x
                past_y = y
        else:
            # it starts to be a new line again
            flag = 1

    else:
        if x:
            x = int(x)
            y = int(y)
            if not finished:
                copied = True
                copied_image = mask.copy()
            if shape is Shape.RECTANGLE:
                cv2.rectangle(copied_image, (past_x, past_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                cv2.circle(copied_image, (past_x, past_y), distance((x, y), (past_x, past_y)), color, thickness)
            if shape is Shape.ELLIPSE:
                cv2.ellipse(copied_image, (past_x, past_y), (abs(x - past_x), abs(y - past_y)),
                            angle(past_x, x, past_y, y), 0., 360, color, thickness)
            if shape is Shape.LINE:
                cv2.line(mask, (past_x, past_y), (x, y), color=color, thickness=thickness)
                past_x = x
                past_y = y

    if finished:
        finished = False
        holding = False
        copied = False
        if x and past_x:
            if shape is Shape.RECTANGLE:
                cv2.rectangle(mask, (past_x, past_y), (x, y), color, thickness)
            if shape is Shape.CIRCLE:
                cv2.circle(mask, (past_x, past_y), distance((x, y), (past_x, past_y)), color, thickness)
            if shape is Shape.ELLIPSE:
                cv2.ellipse(mask, (past_x, past_y), (abs(x + 1 - past_x), abs(y + 1 - past_y)),
                            angle(past_x, x, past_y, y), 0., 360, color, thickness)


def shake_prevention(x, y, past_x, past_y, color, img):
    # print('X     ' + str(x))
    # print('PAST     ' + str(past_x))
    # Distancia ponto atual ao ponto anterior
    if past_x and past_y and x and y:
        # Se a distancia for superior a 50 retorna que é necessário fazer shake prevention caso contrario retorna que não é necessário
        if distance((x, y), (past_x, past_y)) > 100:
            return True
        return False
