import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import cv2
import numpy as np
from face_alignment.headpose import *
import os
from tqdm import tqdm

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, fileList)
    return fileList

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
# cap = cv2.VideoCapture('/home/cmf/t4.mp4')
# try:
#     input_img = io.imread('../test/assets/aflw-test.jpg')
# except FileNotFoundError:
#     input_img = io.imread('test/assets/aflw-test.jpg')
data_dir = '/home/cmf/datasets/align/300W-LP/300W_LP/HELEN'
pic_list = []
GetFileList(data_dir, pic_list)
pic_list = [x for x in pic_list if '.jpg' in x or 'png' in x or 'jpeg' in x]

for pic in tqdm(pic_list):
    img = cv2.imread(pic)
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(input_img)
    if preds is not None:
        print(len(preds))
        preds = preds[0]
        preds = preds.astype(np.int32)
        draw_pose(img, preds)
        draw_distance(img, None, preds)
        draw_expression(img, None, preds)
        # print(preds)
        for i, point in enumerate(preds):
            cv2.circle(img, (point[0], point[1]), 2, (255, 0, 0), -1)
    cv2.imshow('', img)
    cv2.waitKey()

# # 2D-Plot
# plot_style = dict(marker='o',
#                   markersize=4,
#                   linestyle='-',
#                   lw=2)
#
# pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
# pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
#               'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
#               'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
#               'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
#               'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
#               'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
#               'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
#               'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
#               'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
#               }
#
# fig = plt.figure(figsize=plt.figaspect(.5))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(input_img)
#
# for pred_type in pred_types.values():
#     ax.plot(preds[pred_type.slice, 0],
#             preds[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)
#
# ax.axis('off')
#
# # 3D-Plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(preds[:, 0] * 1.2,
#                   preds[:, 1],
#                   preds[:, 2],
#                   c='cyan',
#                   alpha=1.0,
#                   edgecolor='b')
#
# for pred_type in pred_types.values():
#     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
#               preds[pred_type.slice, 1],
#               preds[pred_type.slice, 2], color='blue')
#
# ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
# plt.show()
