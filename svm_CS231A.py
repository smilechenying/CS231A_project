# perform multi-class SVM
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
#from sql_read import *
from sql_read_3dPoint import *
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from matplotlib import cm


image_paths = []
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/TestSet3/train/01-Badshahi Mosque")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/TestSet3/train/02-Dome of the Rock")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/03-Forbidden City Hall/Forbidden City Hall All")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/04-Great Wall/great wall all train")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/05-Azadi Tower/Azadi Tower All Train")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/TestSet3/train/06-Potala Palace")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/07-Qutub Minar/Qutub Minar All Train")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/08-Taj Mahal/Taj Mahal All Train")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/TestSet3/train/09-Temple of Heaven")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/10-Big Ben/Big Ben All Train")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/TestSet3/train/11-Colosseum")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/12-Eiffel Tower/Eiffel Tower All Train")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/13-Golden Gate Bridge/Golden Gate All Train")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/14-Leaning Tower of Pisa/Leaning Tower All Train")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/Images/LincolnMemorial")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/Images/SagradaFamiliaOutside")
image_paths.append("/Users/yingcheny/Downloads/google-images-download-master/downloads/17-Saint Basil Cathedral/Saint Basil Cathedral All")
image_paths.append("/Users/yingcheny/Documents/Ying/CS229-MachineLearning/Winter2016/Project/TestSet3/train/19-Statue of Liberty")


#%%  
des_list = []
fea_agg = np.array([])
des_list_test = []
fea_agg_test = np.array([])
for p in image_paths:
    print('Processing folder: \n')
    print(p)
#    [des_test, features_test, des, features] = read_database(os.path.join(p, 'database.db'))
    [des_test, features_test, des, features] = read_database(p)
    
    des_list.append(des)
    if not fea_agg.any():
        fea_agg = features.copy()
    else:
        fea_agg = np.vstack((fea_agg, features))
    
    des_list_test.append(des_test)
    if not fea_agg_test.any():
        fea_agg_test = features_test.copy()
    else:
        fea_agg_test = np.vstack((fea_agg_test, features_test))

img_num = np.zeros(len(image_paths)).astype('int')
for i in range(0, len(image_paths)):
    img_num[i] = len(des_list[i])
    if i == 0:
        Y = np.zeros(img_num[i])
    else:
        Y = np.hstack((Y, np.zeros(img_num[i]) + i))

img_num_test = np.zeros(len(image_paths)).astype('int')  
for i in range(0, len(image_paths)):
    img_num_test[i] = len(des_list_test[i])
    if i == 0:
        Y_test = np.zeros(img_num_test[i])
    else:
        Y_test = np.hstack((Y_test, np.zeros(img_num_test[i]) + i))
    
#%%    
# Perform k-means clustering
k = 500
voc, variance = kmeans(fea_agg, k, 1) 
#%%
# Calculate the histogram of features
im_features = np.zeros((np.sum(img_num), k), "float32")
for i in xrange(len(image_paths)):
    print(i)
    for j in range(0, img_num[i]):
        words, distance = vq(des_list[i][j],voc)
        for w in words:
            if i ==0:
                cum_num = 0
            else:
                cum_num = np.cumsum(img_num[:i])[-1]
            im_features[j + cum_num][w] += 1
#
# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) 
/ (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)
#
# Train the SVM
clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, 
                fit_intercept=True, loss='hinge')
clf.fit(im_features, Y)

#%%
test_features = np.zeros((np.sum(img_num_test), k), "float32")
for i in xrange(len(image_paths)):
    print(i)
    for j in range(0, img_num_test[i]): 
        words, distance = vq(des_list_test[i][j],voc)
        for w in words:
            if i ==0:
                cum_num = 0
            else:
                cum_num = np.cumsum(img_num_test[:i])[-1]
            test_features[j + cum_num][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / 
                      (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)
predict_c = clf.predict(test_features)
cf_matrix = (confusion_matrix(Y_test, predict_c)/
             np.tile(img_num_test.astype('float'), (18, 1)).T)
predict_diag = np.diag(cf_matrix)/img_num_test.astype('float')
predict_a = clf.predict(im_features)
cf_matrix_a = (confusion_matrix(Y, predict_a)/
               np.tile(img_num.astype('float'), (18, 1)).T)
predict_diag_a = np.diag(cf_matrix_a)/img_num.astype('float')
#joblib.dump((clf, stdSlr, k, voc, im_features, test_features, img_num, img_num_test, Y, Y_test), "bof.pkl", compress=3)  
#joblib.dump((fea_agg, fea_agg_test, des_list, des_list_test), "bof_features.pkl", compress=3)

#%%
clf, stdSlr, k, voc, im_features, test_features, img_num, img_num_test, Y, Y_test = joblib.load("bof_raw.pkl")
fea_agg, fea_agg_test, des_list, des_list_test = joblib.load("bof_features_raw.pkl")

predict_c = clf.predict(test_features)
cf_matrix = confusion_matrix(Y_test, predict_c)/np.tile(img_num_test.astype('float'), (18, 1)).T
predict_diag = np.diag(cf_matrix)/img_num_test.astype('float')
predict_a = clf.predict(im_features)
cf_matrix_a = confusion_matrix(Y, predict_a)/np.tile(img_num.astype('float'), (18, 1)).T
predict_diag_a = np.diag(cf_matrix_a)/img_num.astype('float')


fig, ax = plt.subplots()
cax = ax.imshow(cf_matrix, cmap=cm.coolwarm)
cbar = fig.colorbar(cax)
plt.savefig('test_matrix_2d.png')

fig, ax = plt.subplots()
cax = ax.imshow(cf_matrix_a, cmap=cm.coolwarm)
cbar = fig.colorbar(cax)
plt.savefig('train_matrix_2d.png')

#%%
clf, stdSlr, k, voc, im_features, test_features, img_num, img_num_test, Y, Y_test = joblib.load("bof.pkl")

predict_c = clf.predict(test_features)
cf_matrix = confusion_matrix(Y_test, predict_c)/np.tile(img_num_test.astype('float'), (18, 1)).T
predict_diag = np.diag(cf_matrix)/img_num_test.astype('float')
predict_a = clf.predict(im_features)
cf_matrix_a = confusion_matrix(Y, predict_a)/np.tile(img_num.astype('float'), (18, 1)).T
predict_diag_a = np.diag(cf_matrix_a)/img_num.astype('float')


fig, ax = plt.subplots()
cax = ax.imshow(cf_matrix, cmap=cm.coolwarm)
cbar = fig.colorbar(cax)
plt.savefig('test_matrix_3dPoint.png')

fig, ax = plt.subplots()
cax = ax.imshow(cf_matrix_a, cmap=cm.coolwarm)
cbar = fig.colorbar(cax)
plt.savefig('train_matrix_3dPoint.png')