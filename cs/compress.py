import numpy as np
import cv2
import glob
w,h=28,28
cmp_rate=6
def genBump(bumpSize, RowSize, ColSize, FrameNum):
    BumpTime = np.zeros((RowSize, ColSize), dtype=np.int)
    Mask = np.random.rand(RowSize, ColSize)
    for i in range(FrameNum-bumpSize+1):
        BumpTime[np.logical_and((i / (FrameNum-bumpSize+1)) < Mask, Mask <= ((i+1) / (FrameNum-bumpSize+1)))] = i
    sens_cube = np.zeros((FrameNum, RowSize, ColSize))

    for row in range(RowSize):
        for col in range(ColSize):
            start = BumpTime[row, col]
            sens_cube[start:start + bumpSize, row, col] = 1

    return sens_cube, BumpTime

sens, BumpTime = genBump(3, w,h,cmp_rate)
BumpTime=np.expand_dims(BumpTime,axis=0)
mainf = './OneDrive/Desktop/cs dataset/'
f = '/home/sathya/Desktop/cs_dataset/49503078599@N01_3238848486_5fa56606b7.avi'
cap =  cv2.VideoCapture(f)
vid = []
compressed= []
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(1,1+length):
    # Capture frame-by-frame
#         print((length%cmp_rate==0))

    if(i%cmp_rate==0):
#             print(i)
#             ret, frame = cap.read()
#             vid.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (240, 240)))
#         image_list = np.asarray(image_list)
        ret, frame = cap.read()
        vid.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w, h)))
        image_list = np.asarray(vid)
        image_list = np.reshape(image_list,(cmp_rate,w,h))
        # image_list.shape
        compressed_image = np.multiply(sens,image_list )
        compressed_image = np.sum(compressed_image, 0)/3.
        compressed_image=np.expand_dims(compressed_image,axis=0)
#             print(image_list.shape,BumpTime.shape,compressed_image.shape)
        cs = np.vstack((image_list,BumpTime,compressed_image))
        compressed.append(cs)
        vid=[]

    else:
        ret, frame = cap.read()
        vid.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w, h)))

video = np.asarray(compressed)
np.savez('/home/sathya/Desktop/cs_dataset/'+"vid.npz",video)

import matplotlib.pyplot as plt

plt.imshow(video[0,0,:,:])
plt.show()
plt.imshow(video[0,-1,:,:])
plt.show()