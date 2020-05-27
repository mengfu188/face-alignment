#-*-coding:utf-8-*-


import cv2
import numpy as np


def draw_batch_pose(img, keypoints):
    for i, keypoint in enumerate(keypoints):
        draw_pose(img, keypoint)


def draw_pose(img, keypoints):
    reprojectdst, euler_angle = get_head_pose(keypoints, img)
    pyr = euler_angle.reshape(-1)
    tdx = np.mean(keypoints[0::2])
    tdy = np.mean(keypoints[1::2])
    draw_axis(img, pyr[1], pyr[0], pyr[2],
              tdx, tdy)

    for start, end in line_pairs:
        cv2.line(img, reprojectdst[start], reprojectdst[end], (0, 255, 255), 2)
    cv2.putText(img, f'pitch:{pyr[0]:.2f}', (0, 20 + 0 * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(img, f'yaw:{pyr[1]:.2f}', (0, 20 + 1 * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.putText(img, f'roll:{pyr[2]:.2f}', (0, 20 + 2 * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

import math
from math import cos, sin


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def distance(bbox, landmark):
    if bbox is None:
        bbox = np.array([
            np.min(landmark[0::2]),
            np.min(landmark[1::2]),
            np.max(landmark[0::2]),
            np.max(landmark[1::2]),
        ])
    left = landmark[:, 0::2].min()
    top = landmark[:, 1::2].min()
    right = landmark[:, 0::2].max()
    down = landmark[:, 1::2].max()
    return np.abs(bbox[0] - left), \
           np.abs(bbox[1] - top), \
            np.abs(bbox[2] - right), \
            np.abs(bbox[3] - down)


def draw_distance(img, bbox, landmark, anchor=(0, 100)):
    if bbox is None:
        bbox = np.array([
            np.min(landmark[0::2]),
            np.min(landmark[1::2]),
            np.max(landmark[0::2]),
            np.max(landmark[1::2]),
        ])
    distance_key = ['left', 'top', 'right', 'bottom']
    distance_value = distance(bbox, landmark)
    whwh = [bbox[2] - bbox[0], bbox[3] - bbox[1],
            bbox[2] - bbox[0], bbox[3] - bbox[1], ]
    for j, (key, value) in enumerate(zip(distance_key, distance_value)):
        if key == 'left':
            color = choice_color(value > 10)
        elif key == 'right':
            color = choice_color(value > 10)
        elif key == 'bottom':
            color = choice_color(value > 20)
        else:
            color = (255, 255, 255)

        cv2.putText(img, f'{key}:{value:.2f}', (anchor[0], anchor[1] + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

        if key == 'left':
            color = choice_color(value / whwh[j] > 0.05)
        elif key == 'right':
            color = choice_color(value / whwh[j] > 0.05)
        elif key == 'bottom':
            color = choice_color(value / whwh[j] > 0.10)
        else:
            color = (255, 255, 255)

        cv2.putText(img, f'{key}:{value / whwh[j]:.2f}',
                    (anchor[0] + 100, anchor[1] + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def get_expression(bbox, label, eye_close_thres = 0.02,
                   mouth_close_thres= 0.02,
                   big_mouth_open_thres=0.08,
                   hin = 160, win = 160):
    if bbox is None:
        bbox = np.array([
            np.min(label[0::2]),
            np.min(label[1::2]),
            np.max(label[0::2]),
            np.max(label[1::2]),
        ])
    bbox_height = bbox[3] - bbox[1]
    bbox_width = bbox[2] - bbox[0]
    left_eye_close = np.sqrt(
        np.square(label[37, 0] - label[41, 0]) +
        np.square(label[37, 1] - label[41, 1])) / bbox_height < eye_close_thres \
                     or np.sqrt(np.square(label[38, 0] - label[40, 0]) +
                                np.square(label[38, 1] - label[40, 1])) / bbox_height < eye_close_thres
    right_eye_close = np.sqrt(
        np.square(label[43, 0] - label[47, 0]) +
        np.square(label[43, 1] - label[47, 1])) / bbox_height < eye_close_thres \
                      or np.sqrt(np.square(label[44, 0] - label[46, 0]) +
                                 np.square(label[44, 1] - label[46, 1])) / bbox_height < eye_close_thres

    ###half face
    half_face1 = np.sqrt(np.square(label[36, 0] - label[45, 0]) +
               np.square(label[36, 1] - label[45, 1])) / bbox_width < 0.5
    half_face2 = np.sqrt(np.square(label[62, 0] - label[66, 0]) +
               np.square(label[62, 1] - label[66, 1])) / bbox_height > 0.15
    # big mouth open
    big_mouth_open = np.sqrt(np.square(label[62, 0] - label[66, 0]) +
               np.square(label[62, 1] - label[66, 1])) / hin > big_mouth_open_thres

    return left_eye_close, right_eye_close, half_face1, half_face2, big_mouth_open


def choice_color(value):
    if value == True:
        color = (0, 0, 255)
    elif value == False:
        color = (255, 255, 0)
    else:
        color = (255,255,255)
    return color


def draw_expression(img, bbox, landmark, anchor=(0, 200)):
    expression = get_expression(bbox, landmark)
    expression_key = ['left_eye_close', 'right_eye_close', 'half_face1', 'half_face2', 'big_mouth_open']
    for j, (key, value) in enumerate(zip(expression_key, expression)):
        color = choice_color(value)
        cv2.putText(img, f'{key}:{value}', (anchor[0], anchor[1] + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

# object_pts = np.float32([[6.825897, 6.760612, 4.402142],
#                          [1.330353, 7.122144, 6.903745],
#                          [-1.330353, 7.122144, 6.903745],
#                          [-6.825897, 6.760612, 4.402142],
#                          [5.311432, 5.485328, 3.987654],
#                          [1.789930, 5.393625, 4.413414],
#                          [-1.789930, 5.393625, 4.413414],
#                          [-5.311432, 5.485328, 3.987654],
#                          [2.005628, 1.409845, 6.165652],
#                          [-2.005628, 1.409845, 6.165652],
#                          [2.774015, -2.080775, 5.048531],
#                          [-2.774015, -2.080775, 5.048531],
#                          [0.000000, -3.116408, 6.097667],
#                          [0.000000, -7.415691, 4.070434]])
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652]])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape,img):
    h,w,_=img.shape
    K = [w, 0.0, w//2,
         0.0, w, h//2,
         0.0, 0.0, 1.0]
    # Assuming no lens distortion
    D = [0, 0, 0.0, 0.0, 0]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)



    # image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
    #                         shape[39], shape[42], shape[45], shape[31], shape[35],
    #                         shape[48], shape[54], shape[57], shape[8]])
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
