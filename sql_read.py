import os
import sys
import argparse
import sqlite3
import shutil
import gzip
import numpy as np

def read_database(database_path):

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cameras = {}
    cursor.execute("SELECT camera_id, params FROM cameras;")
    for row in cursor:
        camera_id = row[0]
        params = np.fromstring(row[1], dtype=np.double)
        cameras[camera_id] = params

    images = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        camera_id = row[1]
        image_name = row[2]
        #print(image_id)
        #print "Copying image", image_name
        images[image_id] = (len(images), image_name)

    des_list_train = []
    fea_agg_train = np.array([])
    des_list_test = []
    fea_agg_test = np.array([])
    
    total_num = len(images)    
    train_num = int(total_num*0.8)
    test_num = total_num - train_num
    np.random.seed(1)
    train_idx = np.random.choice(total_num, train_num, replace=False)
    mask = np.ones(total_num, np.bool)
    mask[train_idx] = 0
    counter = 0

    for image_id, (image_idx, image_name) in images.iteritems():
        base_name, ext = os.path.splitext(image_name)

        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                       (image_id,))
        row = next(cursor)
        if row[0] is None:
            keypoints = np.zeros((0, 6), dtype=np.float32)
            descriptors = np.zeros((0, 128), dtype=np.uint8)
        else:
            keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)
            cursor.execute("SELECT data FROM descriptors WHERE image_id=?;",
                           (image_id,))
            row = next(cursor)
            descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)
            descriptors = descriptors.astype('float')
            
        if mask[counter]:
            if not fea_agg_test.any():
                fea_agg_test = descriptors.copy()
            else:
                fea_agg_test = np.vstack((fea_agg_test, descriptors))           
            des_list_test.append(descriptors)
        else:
            if not fea_agg_train.any():
                fea_agg_train = descriptors.copy()
            else:
                fea_agg_train = np.vstack((fea_agg_train, descriptors))           
            des_list_train.append(descriptors)
        counter += 1
    
    return(des_list_test, fea_agg_test, des_list_train, fea_agg_train)
##        key_file_name = key_file_name.replace('/', '_')
#        with open(key_file_name, "w") as fid:
##            fid.write("%d %d\n" % (keypoints.shape[0], descriptors.shape[1]))
#            for r in range(keypoints.shape[0]):
#                fid.write("%f %f " % (keypoints[r, 0], keypoints[r, 1]))
##                fid.write("%f %f %f %f " % (keypoints[r, 0], keypoints[r, 1],
##                                            keypoints[r, 2], keypoints[r, 3]))
#                fid.write(" ".join(map(str, descriptors[r].ravel().tolist())))
#                fid.write("\n")


    cursor.close()
    connection.close()