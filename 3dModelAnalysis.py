#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:38:29 2018

@author: yingcheny
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from mpl_toolkits import mplot3d
import sqlite3


image_path = "Qutub Minar All Train"
out_path = os.path.join(image_path, 'sparse')
filename = os.path.join(out_path, 'images.txt')

# read image feature points
with open(filename) as f:
    lines = f.readlines()
image_name = []
image_id_1 = []
image_key_valid_id = []
image_key_valid = []
image_key_invalid = []
image_quaternion = np.zeros(((len(lines)-4)/2, 8))
for line_id in range(4, len(lines)) :
    if line_id % 2 == 0:
        image_info = lines[line_id].split(' ')
        if len(image_info)>10:
            tmp_name = image_info[-2].split('\n')[0] + ' ' + image_info[-1].split('\n')[0]
        else:
            tmp_name = image_info[-1].split('\n')[0]
        image_name.append(tmp_name)
        image_id_1.append(int(image_info[0]))
        image_quaternion[(line_id-4)/2, :] = np.array(image_info[0:8], dtype=float)
    else:
       image_key_list =  lines[line_id].split(' ')
       image_key_list[-1] = image_key_list[-1].split('\n')[0]
       image_key = np.array(image_key_list, dtype=float).reshape((len(image_key_list)/3), 3)
       image_key_valid.append(image_key[image_key[:, 2] >-1, :-1])
       image_key_invalid.append(image_key[image_key[:, 2] == -1, :-1])
       image_key_valid_id.append(image_key[:, 2])

# sort image_quaternion by camera id
sort_id = image_quaternion[:,0].argsort()
image_quaternion = image_quaternion[sort_id]

##%% read feature and only take the ones for 3D points
#database_path = os.path.join(image_path, 'database.db')
#
#connection = sqlite3.connect(database_path)
#cursor = connection.cursor()
#
#cameras = {}
#cursor.execute("SELECT camera_id, params FROM cameras;")
#for row in cursor:
#    camera_id = row[0]
#    params = np.fromstring(row[1], dtype=np.double)
#    cameras[camera_id] = params
#
#images = {}
#cursor.execute("SELECT image_id, camera_id, name FROM images;")
#image_id_2 = []
#for row in cursor:
#    image_id = row[0]
#    camera_id = row[1]
#    image_name_2 = row[2]
#    images[image_id] = (len(images), image_name)
#    image_id_2.append(image_id)
#
#descriptors_all = []
#keypoints_all = []
#for image_id, (image_idx, image_name_2) in images.iteritems():
#
#    cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
#                   (image_id,))
#    row = next(cursor)
#    if row[0] is None:
#        keypoints = np.zeros((0, 6), dtype=np.float32)
#        descriptors = np.zeros((0, 128), dtype=np.uint8)
#    else:
#        keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)
#        cursor.execute("SELECT data FROM descriptors WHERE image_id=?;",
#                       (image_id,))
#        row = next(cursor)
#        descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)
#    keypoints_all.append(keypoints[:, :2])
#    descriptors_all.append(descriptors)

#%% 
for num_valid in range(0, len(image_id_1)):
    orig_des = descriptors_all[image_id_1[num_valid]-1]
    new_des = np.concatenate((image_key_valid_id[num_valid][image_key_valid_id[num_valid] >-1].reshape(-1,1) ,orig_des[image_key_valid_id[num_valid] >-1]), axis = 1) 
    if num_valid == 0:
        valid_des = new_des.copy()
    else:
        valid_des = np.concatenate((valid_des, new_des), axis = 0)
        
valid_des = valid_des[valid_des[:, 0].argsort(), :]

point_num = int(np.max(valid_des[:, 0]))
point_fea = []
for point_id in range(1, point_num+1):
    valid_point_idex = (valid_des[:, 0] == point_id)
    if valid_point_idex.any():
        point_fea.append(np.mean(valid_des[valid_point_idex, 1:], axis = 0))
point_fea = np.array(point_fea)
np.savetxt(os.path.join(out_path, 'all_features.txt'), point_fea)

#%%
# plot images and feature points
for plot_id in range(0, len(image_name)):
    img = plt.imread(os.path.join(image_path, image_name[plot_id]))
    plt.figure()
    plt.imshow(img)
    plt.plot(image_key_valid[plot_id][:, 0], image_key_valid[plot_id][:, 1], 'r.')
    plt.plot(image_key_invalid[plot_id][:, 0], image_key_invalid[plot_id][:, 1], 'b.')
    filename_noExt = os.path.splitext(image_name[plot_id])[0]
    plt.savefig(os.path.join(out_path, '{}{}.{}'.format(filename_noExt, '_keyPoints', 'png')))
    plt.figure()
    plt.imshow(img)
    plt.plot(image_key_valid[plot_id][:, 0], image_key_valid[plot_id][:, 1], 'b.')
    plt.plot(image_key_invalid[plot_id][:, 0], image_key_invalid[plot_id][:, 1], 'b.')
    filename_noExt = os.path.splitext(image_name[plot_id])[0]
    plt.savefig(os.path.join(out_path, '{}{}.{}'.format(filename_noExt, '_allPoints', 'png')))
    

#%%
# funtion to covert Quaternion to rotation matrix
def qutn2rt(qv):
    rt = np.zeros((3, 4))
    rt[0, 0] = qv[0]**2 + qv[1]**2 - qv[2]**2 - qv[3]**2
    rt[1, 1] = qv[0]**2 - qv[1]**2 + qv[2]**2 - qv[3]**2
    rt[2, 2] = qv[0]**2 - qv[1]**2 - qv[2]**2 + qv[3]**2
    rt[0, 1] = (qv[1]*qv[2] - qv[0]*qv[3])*2
    rt[1, 0] = (qv[1]*qv[2] + qv[0]*qv[3])*2
    rt[0, 2] = (qv[1]*qv[3] + qv[0]*qv[2])*2
    rt[2, 0] = (qv[0]*qv[1] - qv[2]*qv[3])*2
    rt[1, 2] = (qv[2]*qv[3] - qv[0]*qv[1])*2
    rt[2, 1] = (qv[2]*qv[3] + qv[0]*qv[1])*2
    rt[:, :3] = rt[:3, :3]
    rt[:, -1] = qv[4:]
#    rt[:, :3] = rt[:3, :3].T
#    rt[:, -1] = -np.dot(rt[:, :3], qv[4:])
    return rt   
    
# plot 3D plots
filename_3d = os.path.join(out_path, 'points3D.txt')
with open(filename_3d) as f:
    lines_3d = f.readlines()
point_id = []
point_XYZ = np.zeros((len(lines_3d) - 3, 3))
point_RGB = np.zeros((len(lines_3d) - 3, 3))
for line_id in range(3, len(lines_3d)):
    points_info = lines_3d[line_id].split(' ')
    points_info[-1] = points_info[-1].split('\n')[0]
    points_array = np.array(points_info, dtype=float)
    point_id.append(points_array[0])
    point_XYZ[line_id - 3, :] = points_array[1:4]
    point_RGB[line_id - 3, :] = points_array[4:7]


point_h = np.concatenate((point_XYZ, np.ones((point_XYZ.shape[0], 1))), axis=1)

# read estimated camera intrinsic parameters
filename_camera = os.path.join(out_path, 'cameras.txt')
with open(filename_camera) as f:
    lines_camera = f.readlines()
camera_param = np.zeros((len(lines_camera) - 3, 4))
for line_id in range(3, len(lines_camera)):
    camera_info = lines_camera[line_id].split(' ')
    camera_info[-1] = camera_info[-1].split('\n')[0]
    camera_param[line_id - 3, :] = np.array(camera_info[4:8], dtype=float)

# reconstruct 3d points to 2d
camera_M_in = np.zeros((3, 3))
camera_M_in[2, 2] = 1.0
for point_id in range(0, image_quaternion.shape[0]):
#for point_id in range(0, 1):
    rt = qutn2rt(image_quaternion[point_id,1:])
    fl = camera_param[point_id, 0]
    camera_M_in[0, 0] = fl
    camera_M_in[1, 1] = fl
    point_2d = np.dot(rt, point_h.T)
    point_2d_norm = point_2d / np.tile(point_2d[-1, :], (3, 1))    
    k = camera_param[point_id, -1]
    principal_point = camera_param[point_id, 1:3]
    
    u2 = point_2d_norm[0, :] ** 2
    v2 = point_2d_norm[1, :] ** 2
    r2 = point_2d_norm[0, : ]*point_2d_norm[1, :]
    radial = k * r2
    point_h_radial = np.zeros((point_2d_norm.shape[0], point_2d_norm.shape[1]))
    point_h_radial[0, :] = point_2d_norm[0, :] * radial + point_2d_norm[0, :]
    point_h_radial[1, :] = point_2d_norm[1, :] * radial + point_2d_norm[1, :]
    point_h_radial[2, :] = point_2d_norm[2, :].copy()
#    point_2d_img = np.dot(camera_M_in, point_h_radial)
#    point_2d_out = point_2d_img[:2, :] + np.tile(principal_point, (point_2d_norm.shape[1], 1)).T
    
    camera_M_in[0,2] = principal_point[0]
    camera_M_in[1,2] = principal_point[1]
    point_2d_out = np.dot(camera_M_in, point_h_radial)
    
    
    img = plt.imread(os.path.join(image_path, image_name[sort_id[point_id]]))
    plt.figure()
    plt.imshow(img)
    plt.plot(image_key_valid[sort_id[point_id]][:, 0], image_key_valid[sort_id[point_id]][:, 1], 'r.')
    #plt.plot(image_key_invalid[point_id][:, 0], image_key_invalid[point_id][:, 1], 'b.')
    plt.plot(point_2d_out[0, :], point_2d_out[1, :], 'b+')
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)
    filename_noExt = os.path.splitext(image_name[sort_id[point_id]])[0]
    plt.savefig(os.path.join(out_path, '{}{}.{}'.format(filename_noExt, '_project3D', 'png')))
#    
##plt.figure(figsize=[10,8])
##ax = plt.axes(projection='3d')
##ax.scatter3D(point_XYZ[:, 0], point_XYZ[:, 1], point_XYZ[:, 2], point_RGB)
##plt.show()
#
##%%
## plot reprojected points
