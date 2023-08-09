#!/usr/bin/env python
# coding: utf-8

# # import numpy as np
# import cv2
# 
# color = [255,0,0]  #파란색
# pixel = np.uint8([[color]]) # 한 픽셀로 구성된 이미지로 변환
# 
# # BGR -> HSV
# hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
# print(hsv, 'shape:', hsv.shape )
# 
# # 픽셀값만 가져오기 
# hsv = hsv[0][0]
# 
# print("bgr: ", color)
# print("hsv: ", hsv) # +_ 10

# # HSV
# 1. H(색상, Color, Hue): 일반적으로 부르는 색 이름
# 2. S(채도, Saturation): 색의 선명함 정도 -> 채도가 0이면 무채색
# 3. V(명도, Brightness, Value): 색의 밝고 어두움

# # 특정 색 추출

# In[1]:


#웹캠으로 실시간으로 특정색 물체 추적하기
import cv2 as cv
import numpy as np


hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0


def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환
    if event == cv.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        threshold = cv.getTrackbarPos('threshold', 'img_result')
        
        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위 정하기
        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-10+180, threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], threshold, threshold])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

cv.namedWindow('img_result')
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'img_result', 30)

cap = cv.VideoCapture(0)

while(True):
    ret,img_color = cap.read()
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    kernel = np.ones((11,11), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)
    #cv2.bitwise_and: 입력 이미지 간의 각 픽셀 값을 비교하고, 두 이미지의 해당 픽셀 값이 모두 0이 아닌 경우에만 결과 이미지인 dst의 해당 픽셀 값을 1로 설정-> 두 이미지가 겹치는 영역만을 유지, 나머지 부분은 모두 검정색으로 만들어서 마스킹하는 효과

    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x,y,width,height,area = stats[idx]
        centerX,centerY = int(centroid[0]), int(centroid[1])
        print(centerX, centerY)

        if area > 50:
            cv.circle(img_color, (centerX, centerY), 10, (0,0,255), 10)
            cv.rectangle(img_color, (x,y), (x+width,y+height), (0,0,255))

    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)


    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()


# # 영상에서 R, G, B 추출하기

# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)       #카메라 모듈 사용.

while(1):
    ret, frame = cap.read()     #카메라 모듈 연속프레임 읽기

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    #BGR을 HSV로 변환

    lower_blue = np.array([100,100,120])       #파랑색 범위
    upper_blue = np.array([150,255,255])

    lower_green = np.array([50, 150, 50])      #초록색 범위
    upper_green = np.array([80, 255, 255])

    lower_red = np.array([150, 50, 50])        #빨강색 범위
    upper_red = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)     #100<->150 Hue(색상) 영역을 지정.
    mask1 = cv2.inRange(hsv, lower_green, upper_green)  #영역 이하는 모두 날림 검정. 그 이상은 모두 흰색 두개로 Mask를 씌움.
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(frame, frame, mask=mask)      #흰색 영역에 파랑색 마스크를 씌워줌.
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)    #흰색 영역에 초록색 마스크를 씌워줌.
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)    #흰색 영역에 빨강색 마스크를 씌워줌.

    cv2.imshow('frame',frame)       #원본 영상
    cv2.imshow('Blue', res)           #마스크 위에 파랑색을 씌운 것을 보여줌.
    cv2.imshow('Green', res1)          #마스크 위에 초록색을 씌운 것을 보여줌.
    cv2.imshow('red', res2)          #마스크 위에 빨강색을 씌운 것을 보여줌.

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


# # 영상에서 노란색 추출하기(변형)

# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)       #카메라 모듈 사용.

while(1):
    ret, frame = cap.read()     #카메라 모듈 연속프레임 읽기

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    #BGR을 HSV로 변환
                           #색상 채도 명도 
    lower_yellow = np.array([20,20,100])          #노란색 범위
    upper_yellow = np.array([40,255,255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 

    res = cv2.bitwise_and(frame, frame, mask=mask)      

    cv2.imshow('frame',frame)       
    cv2.imshow('Yellow', res)           

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


# # Trackbar

# In[ ]:


import time
import cv2
import numpy as np

def onChange(x):
    pass

def setting_bar():
    cv2.namedWindow('HSV_settings')

    cv2.createTrackbar('H_MAX', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('H_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('H_MIN', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('H_MIN', 'HSV_settings', 0)
    cv2.createTrackbar('S_MAX', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('S_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('S_MIN', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('S_MIN', 'HSV_settings', 0)
    cv2.createTrackbar('V_MAX', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('V_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('V_MIN', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('V_MIN', 'HSV_settings', 0)

setting_bar()


def showcam():
    try:
        print('open cam')
        cap = cv2.VideoCapture(0)
    except:
        print('Not working')
        return
    cap.set(3, 480)
    cap.set(4, 320)

    while True:
        ret, frame = cap.read()
        H_MAX = cv2.getTrackbarPos('H_MAX', 'HSV_settings')
        H_MIN = cv2.getTrackbarPos('H_MIN', 'HSV_settings')
        S_MAX = cv2.getTrackbarPos('S_MAX', 'HSV_settings')
        S_MIN = cv2.getTrackbarPos('S_MIN', 'HSV_settings')
        V_MAX = cv2.getTrackbarPos('V_MAX', 'HSV_settings')
        V_MIN = cv2.getTrackbarPos('V_MIN', 'HSV_settings')
        lower = np.array([H_MIN, S_MIN, V_MIN])
        higher = np.array([H_MAX, S_MAX, V_MAX])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Gmask = cv2.inRange(hsv, lower, higher)
        G = cv2.bitwise_and(frame, frame, mask = Gmask)
        if not ret:
            print('error')
            break
        cv2.imshow('cam_load',frame)
        cv2.imshow('G',G)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

showcam()


# # Trackbar 사용으로 더 정확한 세팅값 확인

# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)       

while(1):
    ret, frame = cap.read()     

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
                           #색상 채도 명도 
    lower_yellow = np.array([20,50,100])          
    upper_yellow = np.array([45,100,200])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 

    res = cv2.bitwise_and(frame, frame, mask=mask)      

    cv2.imshow('frame',frame)       
    cv2.imshow('Yellow', res)           

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


# # 실시간으로 노란색물체 추적

# In[ ]:


import cv2 as cv
import numpy as np

# Define the HSV range for yellow color
#lower_yellow = np.array([20, 100, 100])
#upper_yellow = np.array([40, 255, 255])
lower_yellow = np.array([20, 50, 100])
upper_yellow = np.array([45, 100, 200])


def nothing(x):
    pass

cv.namedWindow('img_result')
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'img_result', 30)

cap = cv.VideoCapture(0)

while True:
    ret, img_color = cap.read()
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

    # Convert the original image to HSV
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # Create a mask for the yellow color range
    img_mask = cv.inRange(img_hsv, lower_yellow, upper_yellow)

    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

    # Apply the mask to the original image to get the result
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        print(centerX, centerY)

        if area > 50:
            cv.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
            cv.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))

    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)

    # Press ESC key to exit
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()


# # CNN으로 점자 블록 인식 도전

# In[ ]:


#이미지들을 읽어 들일 경로 지정
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf

import os
import PIL
import shutil
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#훈련용 O/X 이미지 경로
train_dir = 'train'  #폴더 이름
train_line_dir = 'train_line'
train_dot_dir = 'train_dot'
#print(train_o_dir, train_x_dir)

#검증용 O/X 이미지 경로
validation_dir = 'validation'  #폴더 이름
validation_line_dir = 'validation_line'
validation_dot_dir = 'validation_dot'
#print(validation_o_dir, validation_x_dir)

#테스트용 O/X 이미지 경로
test_dir = 'testtest'  #폴더 이름
test_line_dir = 'test_line'
test_dot_dir = 'test_dot'
#print(test_o_dir, test_x_dir)

#훈련용 이미지 파일 이름 조회
train_line_fname = [os.path.join(train_line_dir, file_name) for file_name in os.listdir(train_line_dir)]
train_dot_fname = [os.path.join(train_dot_dir, file_name) for file_name in os.listdir(train_dot_dir)]
#print(train_o_fname, train_x_fname)
#os.listdir(): ()에 있는 모든 파일과 디렉토리의 이름을 포함
#os.path.join(train_o_dir, file_name): 파일의 전체 경로 생성, 운영체제에 맞게 디렉토리 경로 연결
#for file_name in os.listdir(train_o_dir): os,listdir(trian_o_dir)에서 반환된 각 파일과 디렉토리 이름을 순회


#각 디렉토리에 저장되어 있는 이미지 파일의 수
print('Total training line images :', len(os.listdir(train_line_dir)))
print('Total training dot images :', len(os.listdir(train_dot_dir)))

print('Total validation line images :', len(os.listdir(validation_line_dir)))
print('Total validation dot images :', len(os.listdir(validation_dot_dir)))

print('Total test line images :', len(os.listdir(test_line_dir)))
print('Total test dot images :', len(os.listdir(test_dot_dir)))
#len(): 인자로 전달된 객체의 길이 또는 요소의 개수 반환

#이미지 확인
nrows, ncols = 4, 4  #가로. 세로 길이
pic_index = 0

fig = plt.gcf()  #현재 활성화된 figure 객체 가져오기
fig.set_size_inches(ncols*3, nrows*3)  #그리드의 크기 설정

pic_index += 8  #pic_index 값 업데이트

next_line_pix = [os.path.join(train_line_dir, file_name) for file_name in os.listdir(train_line_dir)[pic_index-8:pic_index]]
next_dot_pix = [os.path.join(train_dot_dir, file_name) for file_name in os.listdir(train_dot_dir)[pic_index-8:pic_index]]
#enumerate(): 리스트의 인덱스와 해당 인덱스의 이미지 파일 경로를 순회
#인덱스: 리스트나 배열의 각 요소가 위치한 위치를 나타내는 숫자(0부터 시작)
for i, img_path in enumerate(next_line_pix + next_dot_pix):  #i: 현재 요소의 인덱스, img_path: 현재 요소의 값(이미지 파일 경로)
    sp = plt.subplot(nrows, ncols, i+1)  #i+1: 인덱스를 1부터 시작하는 방식으로 변환
    sp.axis('OFF')
    
    img = mping.imread(img_path)  #저장된 이미지 파일을 읽어들임
    plt.imshow(img)  #이미지가 현재 서브플롯에 표시
    
plt.show()
#mping.imread(): 다양한 이미지 포멧(PNG, JPEG, BMP, TIFF 등) 지원, 기본적으로 RGB 색상 순서로 이미지 반환
#cv2.imread(): 다양한 이미지 포멧(JPEG, PNG, BMP 등) 지원 하지만 일부는 지원 X, BGR 색상 순서로 이미지 반환

#Image augmentation: 이미지 데이터 전처리
#train셋에만 적용
train_datagen = ImageDataGenerator(rescale = 1./255, # 모든 이미지 원소값들(각 픽셀의 색상 강도)을 255로 나누기 -> 0과 1사이로 변환(정규화)
                                   rotation_range=90, # 0~25도 사이에서 임의의 각도로 원본이미지를 회전
                                   width_shift_range=0.05, # 0.05범위 내에서 임의의 값만큼 임의의 방향으로 좌우 이동
                                   height_shift_range=0.05, # 0.05범위 내에서 임의의 값만큼 임의의 방향으로 상하 이동
                                   zoom_range=0.2, # (1-0.2)~(1+0.2) => 0.8~1.2 사이에서 임의의 수치만큼 확대/축소
                                   horizontal_flip=True, # 좌우로 뒤집기                                   
                                   vertical_flip=True,
                                   fill_mode='nearest'  #이미지 데이터 증강 중에 새로운 픽셀이 생성되거나 이미지가 이동되었을 때 가장 가까운 픽셀의 값을 사용하여 빈 공간을 채움
                                   ) 
# validation 및 test 이미지는 augmentation을 적용하지 않는다;
# 모델 성능을 평가할 때에는 이미지 원본을 사용 (rescale만 진행)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# flow_from_directory() 메서드를 이용해서 훈련과 테스트에 사용될 이미지 데이터를 만들기
# 변환된 이미지 데이터 생성
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    batch_size=16, # 한번에 변환된 이미지 16개씩 만들어라 라는 것
                                                    color_mode='grayscale', # 흑백 이미지 처리
                                                    class_mode='binary', 
                                                    target_size=(150,150)) # target_size에 맞춰서 이미지의 크기가 조절된다

validation_generator = validation_datagen.flow_from_directory(validation_dir, 
                                                              batch_size=4, 
                                                              color_mode='grayscale',
                                                              class_mode='binary', 
                                                              target_size=(150,150))

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=4,
                                                  color_mode='grayscale',
                                                  class_mode='binary',
                                                  target_size=(150,150))

# 참고로, generator 생성시 batch_size x steps_per_epoch (model fit에서) <= 훈련 샘플 수 보다 작거나 같아야 한다.
#steps_per_epoch: 총 데이터 샘플 수/배치 크기
#steps_per_epoch: 배치를 구성하는 과정을 전체 데이터셋에 대해 몇 번 반복하는지 결정하는 매개변수 -> steps_per_epoch= generator/batch_size


# In[ ]:


# class 확인
train_generator.class_indices


# 모델 구성

# In[ ]:


import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()


# 모델 학습

# In[ ]:


#모델 컴파일
from tensorflow.keras.optimizers import RMSprop

# compile() 메서드를 이용해서 손실 함수 (loss function)와 옵티마이저 (optimizer)를 지정
model.compile(optimizer=RMSprop(learning_rate=0.001), # 옵티마이저로는 RMSprop(학습률을 적절하게 변화시킴) 사용
              loss='binary_crossentropy', # 손실 함수로 ‘binary_crossentropy’ 사용, 모델의 예측과 실제 레이블 사이의 차이 측정
              metrics= ['accuracy'])
#손실 함수: 모델의 출력과 실제 타깃 값 사이의 차이를 측정하여 모델의 성능을 평가하는 함수
#옵티마이저: 모델의 가중치를 업데이트하는 알고리즘, 손실 함수를 최소화 하는 방향으로 모델을 조정

# 모델 훈련
history = model.fit_generator(train_generator, 
                              validation_data=validation_generator, 
                              steps_per_epoch=4, # 한 번의 epoch에서 훈련에 사용할 batch의 개수 지정 -> generator 4번
                              epochs=200, # epochs: 데이터셋을 한 번 훈련하는 과정
                              validation_steps=4, # 한 번의 epoch가 끝날 때, 검증에 사용되는 batch의 개수를 지정 -> 4개의 배치로 나누어 모델을 4번 평가
                              verbose=2)
#validation_steps는 보통 자신이 원하는 이미지 수에 flow할 때 지정한 batchsize로 나눈 값을 validation_steps로 지정


# In[ ]:


# 모델 성능 평가
model.evaluate(train_generator)
model.evaluate(validation_generator)


# 테스트 평가

# In[ ]:


# 이제 테스트 이미지 분류
import numpy as np
from PIL import Image
import os

# 테스트용 line 이미지 경로 설정 
test_line_dir = 'test_line'
test_line_filenames = [os.path.join(test_line_dir, file_name) for file_name in os.listdir(test_line_dir)]

# 테스트용 dot 이미지 경로 설정 
test_dot_dir = 'test_dot'
test_dot_filenames = [os.path.join(test_dot_dir, file_name) for file_name in os.listdir(test_dot_dir)]

# line, dot을 key로, 이미지 파일 이름들을 value로 dictionary 생성
dic_ld_filenames = {}  #두 개의 키 'O', 'X'를 가지는 딕셔너리
dic_ld_filenames['line'] = test_line_filenames
dic_ld_filenames['dot'] = test_dot_filenames
#키(key): 딕셔너리에서 각 항목을 식별하는 역할을 하는 요소

# line/dot 분류 테스트
for ld, filenames in dic_ld_filenames.items():
    fig = plt.figure(figsize=(16,10))
    rows, cols = 1, 6
    for i, fn in enumerate(filenames):
        test_img = Image.open(fn).convert('L').resize((150, 150))  #이미지 파일을 연 후 흑백으로 변환하고 크기를 (150, 150)으로 조정       
        x = image.img_to_array(test_img)  #img_to_array: PIL 이미지 객체를 넘파이 배열로 변환
        x = np.expand_dims(x, axis=0)  #expand_dims: 배열의 차원 확장 -> 입력 데이트를 배치 형태로 만들기 위해 차원 추가
        images = np.vstack([x])  #vstack: 배열들을 수직으로 쌓아 하나의 배열로 만듦

        classes = model.predict(images, batch_size=10)  #입력 데이터에 대해 모델의 예측을 수행하고 각 입력에 대한 예측 결과를 반환
        #classes: 반환값, images: 예측하고자 하는 이미지 데이터를 담고 있는 배열, batch_size: 한 번에 처리할 배치의 크기
        
        fig.add_subplot(rows, cols, i+1)  #이미지 시각화
        if classes[0]==0:  #해당 이미지가 'dot' 인 경우
            plt.title(fn + " is dot")
            plt.axis('off')
            plt.imshow(test_img, cmap='gray')

        else:  #해당 이미지가 'line' 인 경우
            plt.title(fn + " is line")
            plt.axis('off')
            plt.imshow(test_img, cmap='gray')
    plt.show();


# In[ ]:


# 모델 성능 평가
model.evaluate(test_generator)


# # ----------------------------------------------------

# # 사물인식(Object Detection) 예제

# In[1]:


from IPython.display import Image, display

path = "cat_dog.jpeg"  #사진 파일의 디렉토리
display(Image(filename = path))


# In[3]:


import cv2
import cvlib as cv 
from IPython.display import Image, display

img = cv2.imread(path)  #이미지 파일 불러오기
path = "cat_dog.jpeg"  #사진 파일의 디렉토리
display(Image(filename = path))

conf = 0.5  #사물 인식을 진행할 confidence의 역치 값
model_name = "yolov3"  #사물을 인식할 모델 이름

result = cv.detect_common_objects(img, confidence=conf, model=model_name)

output_path = "cat_dog.jpeg"  #결과가 반영된 이미지 파일 저장 디렉토리

result_img = cv.object_detection.draw_bbox(img, *result)  #result 결과를 이미지에 반영
cv2.imwrite(output_path, result_img)  #반영된 이미지 파일 저장
display(Image(filename = output_path))  #이미지 출력


# # yolov5 해시 매칭

# In[4]:


import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from gtts import gTTS
import playsound
import os


# TTS
def speak(text):
    tts = gTTS(text=text, lang='ko')
    # 파일 이름 설정
    filename = 'mmyeong.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)  # 한 객체당 mp3파일을 계속 만들어서 한객체 생성후 삭제 -> 객체 생성 삭제 반복


def image_min_max(image, k=3):  # k-means 최대 최소 값 구하는 함수
    image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width 통합

    clt = KMeans(n_clusters=k)  # k개의 데이터 평균을 만들어 데이터를 clustering하는 알고리즘
    clt.fit(image)  # 컬러값

    n1 = clt.cluster_centers_  # 컬러값 numpy 배열

    min_BGR = np.apply_along_axis(lambda a: np.min(a), 0, n1)  # 최소값
    max_BGR = np.apply_along_axis(lambda a: np.max(a), 0, n1)  # 최대값

    # print("\nmin", min_BGR) #클러스터 최대최소값 확인
    # print("max", max_BGR)

    return min_BGR, max_BGR


def image_Binarization(image, min, max):  # 원본 이미지에 적용하는 함수
    dst = cv2.inRange(image, min, max)  # 추출된 RGB값 최소 최대 범위 지정

    kernel = np.ones((33, 33), np.uint8)

    # 모폴로지 노이즈 제거
    closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    avg = dst.mean()

    gray = cv2.resize(opened, (5, 5))  # 5x5 크기로 축소
    hash = 1 * (gray > avg)  # True False 를 0과 1로 변환

    sum_width = []
    sum_length = []
    for a in hash[3, :]:
        sum_width.append(a)  # 3행 추출
    for a in hash[:, 3]:
        sum_length.append(a)  # 3열 추출

    if sum(sum_width) + sum(sum_length) == 10:  # 3행 3열 합이 10이면 사거리
        print("전방에 사거리블록이 있습니다.")
        # speak("전방에 사거리블록이 있습니다.")
    elif sum(sum_width) + sum(sum_length) == 8:  # 3행 3열 합이 8이면 삼거리
        print("전방에 사거리블록이 있습니다.")
        # speak("전방에 삼거리블록이 있습니다.")

    plt.imshow(opened, cmap='gray')
    plt.savefig('result.jpg')  # 이미지 파일 저장


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        x = int(xyxy[0])  # x좌표
                        y = int(xyxy[1])  # y좌표
                        w = int(xyxy[2])  # 넓이
                        h = int(xyxy[3])  # 높이
                        middle_x = int((x + w) / 2)  # x좌표 시작점
                        middle_y = int((y + h) / 2)  # y좌표 시작점

                        if names[int(cls)] == "ThreeWayBlock" or names[int(cls)] == "IntersectionBlock" and conf >= 0.6:
                            a = 10
                            region = im0[int(middle_y) - a:int(middle_y) + a, int(middle_x) - a:int(middle_x) + a]
                            min, max = image_min_max(region)
                            img_b_box = im0[y:h, x:w]  # 바운딩 박스 좌표
                            image_Binarization(img_b_box, min, max)  # 바운딩 박스만 이진화 진행
                        elif names[int(cls)] == "GoStraight":  # 직진
                            print('직진 블록이 있습니다.')
                            # speak('직진 블록이 있습니다.')
                        elif names[int(cls)] == "Stop":  # 정지
                            print('정지 블록이 있습니다.')
                            # speak('정지 블록이 있습니다.')

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

    # defalut 값 설정으로 boundingbox 출력 범위 제한
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


# # yolov5 꼭짓점 좌표 분석

# In[5]:


import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math


###라벨링 함수###
def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)  # 바운딩 박스 좌표 추출
    pt1 = (x, y + 630)
    pt2 = (x + w, y + h)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))  # 텍스트 기입


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(
                            cls)], line_thickness=3)  # plot the bounding boxes

                        if names[int(cls)] == 'ThreeWayBlock' and conf >= 0.6:
                            ########################################################################################################################################
                            x1, y1, x2, y2 = int(xyxy[0].item()), int(xyxy[1].item()), int(xyxy[2].item()), int(
                                xyxy[3].item())  # bounding box

                            middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2  # creating middle point in bounding box

                            x11, x22, y11, y22 = int(middle_x) - 30, int(middle_x) + 30, int(middle_y) - 30, int(
                                middle_y) + 30  # small rectangular box from middle point

                            img = im0
                            if img is None:
                                print('Image load failed!')
                                return

                            region = im0[y11:y22,
                                     x11:x22]  # crop the region created from small rectangular from middle point
                            b, g, r = np.mean(region, axis=(0, 1)).round(
                                2)  # generating the average values of bgr in selected region
                            # b, g, r = np.median(region, axis=(0, 1)).round(2)
                            # print([b,g,r])

                            kernel = np.ones((33, 33), np.uint8)  # 커널값
                            # creating range from average bgr

                            lower = (b - 10, g - 10, r - 10)  # BGR minimum 범위
                            higher = (b + 10, g + 10, r + 10)  # BGR Maximum 범위
                            cropped_img = im0[y1:y2, x1:x2]  # 관심영역 추출
                            dst = cv2.inRange(cropped_img, lower, higher)  # 관심영역안의 원하는 색상범위로 이진화처리
                            closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)  # 모폴로지 클로즈연산
                            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 모폴로지 오프닝 연산

                            ################################contours####################################################
                            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_NONE)  # findContours 함수로 외곽선 추출
                            for pts in contours:  # 노이즈 예외처리
                                if cv2.contourArea(pts) < 2000:
                                    continue
                                approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.01, True)  # 외곽선 근사화 처리
                                # print("approx : ", approx) #근사화 값 print로 확인용

                                vtc = len(approx)  # 꼭짓점 개수 추출
                                for i in range(approx.shape[0]):
                                    pt = (int(approx[i, 0, 0]),  # 꼭짓점 개수만큼 for문을 돌면서 좌표값을 pt에 저장
                                          int(approx[i, 0, 1]))
                                    cropped_img = im0[y1:y2, x1:x2]  # 욜로 바운딩박스에서 추출한 관심영역 좌표값 추출
                                    cv2.circle(cropped_img, pt, 5, (0, 0, 255), 2)  # 추출한 꼭짓점을 시각화하기위해 Circle함수 사용
                                    # cv2.circle(opened, pt, 5, (0, 0, 255), 2)
                                if vtc == 4:
                                    # setLabel(img, pts, 'GoStraight')
                                    setLabel(img, pts, '')
                                elif vtc == 12:  # 사거리의 경우 꼿짓점 12
                                    print("사거리입니다.")
                                    # speak('사거리입니다.')#TTS 사용시 출력할 Speak
                                    # setLabel(img, pts, 'ThreeWayBlock')
                                    setLabel(img, pts, '')
                                elif vtc == 8:  # 삼거리의 경우 꼭짓점 8
                                    print("삼거리입니다.")
                                    # speak('삼거리입니다.')#TTS 사용시 출력할 Speak
                                    # setLabel(img, pts, 'ThreeWayBlock')
                                    setLabel(img, pts, '')
                            print(vtc)
                            Canny_HSV = cv2.Canny(opened, 50, 150)  # canny edge

                            # cv2.imshow('Canny_', Canny_HSV)
                            # cv2.imshow('Gaussian',opened)
                            cv2.imshow('img', img)
                            cv2.waitKey()
                            cv2.destroyAllWindows()
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)# plot the bounding boxes

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:  # image 저장 함수
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')  # 한 프레임당 처리 시간 print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt',
                        help='model.pt path(s)')  # last.pt : 학습된 YOLOv5 모델명
    parser.add_argument('--source', type=str, default='example0526.jpg',
                        help='source')  # default에서 '0' 으로 할시 Webcam으로 실시간 영상처리 가능
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


# # 객체탐지 알고리즘 구현(yolov2)

# In[ ]:


import cv2
import numpy as np
import cv2 as cv

# 웹캠 신호 받기
VideoSignal = cv2.VideoCapture(0)
# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

# YOLO NETWORK 재구성
classes = []
with open("yolo.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

while True:
    # 웹캠 프레임
    ret, frame = VideoSignal.read()
    h, w, c = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#blob: 이미지에서 특징을 잡아내고 크기를 조정하는데 사용
#cv2.dnn.blobFromImage(): 딥러닝 모델에 이미지를 입력으로 전달하기 위해 이미지 전처리 하는 함수
#img: 입력 이미지
#scalefactor: 입력 이미지의 픽셀값을 조절하는 비율(정규화)
#size: 변환된 이미지의 크기를 지정하는 튜플
#mean: 입력 이미지의 채널별 평균값을 지정하는 튜플
#swapRB: R, G, B 채널 순서를 변경하는 플레그(True로 설정하면 opencv의 기본적인 BGR 채널 순서를 RGB로 변경)
#crop: 이미지를 중앙을 기준으로 크기에 맞게 자르는지 여부를 나타내는 플레그(False로 설정하면 이미지가 중앙에 위치하도록 조정)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)
#탐지된 객체들에 대한 정보를 저장할 빈 리스트들
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]  #각 박스에 대한 클래스 확률 추출/ [center_x, center_y, width, height, confidence, class_probabilities]
            class_id = np.argmax(scores)  #클래스 확률 중 가장 높은 값을 가지는 인덱스 찾아 저장
            confidence = scores[class_id]  #탐지된 객체의 신뢰도 저장

            if confidence > 0.5:
                #객체 감지
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                #박스 좌표 계산
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

#겹치는 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 이미지에 입력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv3", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()


# # 객체탐지 알고리즘 구현(yolov3)

# In[7]:


import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  #미리 학습된 yolo모델 불러오기
classes = []  #클래스들을 저장하는 리스트
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread("sample.jpg")
img = cv2.resize(img, None, fx=1, fy=1)
height, width, channels = img.shape

#객체 감지
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#(320, 320): 빠르지만 정확도 낮음
#(609, 609): 느리지만 정확도 높음
#(416, 416): 반반
net.setInput(blob)  #전처리된 이미지를 yolo 모델에 입력으로 제공
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]  #신뢰도 -> 1로 갈수록 정확도 높음
        if confidence > 0.5:
            #객체 인식
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            #박스 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  #같은 객체에 생기는 여러개의 박스 제거

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 자동차 인식

# In[7]:


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

box = pd.read_csv('train_solution_bounding_boxes (1).csv')
#pd: pandas 라이브러리의 약칭, 데이터 분석과 조작을 위한 강력한 라이브러리
#read_csv: CSV 파일을 읽어와 DataFrame으로 변환하는 기능 제공
box


# In[8]:


sample = cv2.imread('training_images/vid_4_1000.jpg')
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)  #RGB 채널 순서로 변경
point = box.iloc[0]  #DataFrame의 첫 번째 행
pt1 = (int(point['xmin']), int(point['ymax']))
pt2 = (int(point['xmax']), int(point['ymin']))  
cv2.rectangle(sample, pt1, pt2, color=(255,0,0), thickness=2)
plt.imshow(sample)


# In[9]:


sample = cv2.imread('training_images/vid_4_10000.jpg')
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
point = box.iloc[1]  #DataFrame의 두 번째 행
pt1 = (int(point['xmin']), int(point['ymax']))
pt2 = (int(point['xmax']), int(point['ymin']))
cv2.rectangle(sample, pt1, pt2, color=(255,0,0), thickness=2)
plt.imshow(sample)


# In[10]:


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 이미지 가져오기
img = cv2.imread('training_images/vid_4_10000.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 1/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
#blob: 이미지에서 특징을 잡아내고 크기를 조정하는데 사용
#cv2.dnn.blobFromImage(): 딥러닝 모델에 이미지를 입력으로 전달하기 위해 이미지 전처리 하는 함수
#img: 입력 이미지
#scalefactor: 입력 이미지의 픽셀값을 조절하는 비율(정규화)
#size: 변환된 이미지의 크기를 지정하는 튜플
#mean: 입력 이미지의 채널별 평균값을 지정하는 튜플
#swapRB: R, G, B 채널 순서를 변경하는 플레그(True로 설정하면 opencv의 기본적인 BGR 채널 순서를 RGB로 변경)
#crop: 이미지를 중앙을 기준으로 크기에 맞게 자르는지 여부를 나타내는 플레그(False로 설정하면 이미지가 중앙에 위치하도록 조정)
net.setInput(blob)

# outs: 출력으로 탐지된 개체에 대한 모든 정보와 위치를 제공
outs = net.forward(output_layers)

# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #객체 감지
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            #박스 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#box: 감지된 개체를 둘러싼 사각형의 좌표
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    print(x, y, w, h)
    label = str(classes[class_ids[i]])  #label: 감지된 물체의 이름
    confidence = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(img, (x, y), ((x+w), (y+h)), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0, 255, 0), 2)

plt.imshow(img)


# In[16]:


def predict_yolo(img_path):
  # 이미지 가져오기
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #RGB 형식으로 변환
  height, width, channels = img.shape

#이미지 전처리
  blob = cv2.dnn.blobFromImage(img, 1/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
  net.setInput(blob) 
  outs = net.forward(output_layers)

  class_ids = []
  confidences = []
  boxes = []
  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)  #객체의 클래스를 나타내는 인덱스
          confidence = scores[class_id]
          if confidence > 0.5:
              #객체 감지
              center_x = int(detection[0] * width)
              center_y = int(detection[1] * height)
              w = int(detection[2] * width)
              h = int(detection[3] * height)
              #박스 좌표
              x = int(center_x - w / 2)
              y = int(center_y - h / 2)
              boxes.append([x, y, w, h])  #좌표 정보
              confidences.append(float(confidence))
              class_ids.append(class_id)

#겹치는 박스 제거
  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  font = cv2.FONT_HERSHEY_PLAIN  #평범평범
  colors = np.random.uniform(0, 255, size=(len(boxes), 3))  #[0, 255] 범위에서 무작위로 색상을 생성
  if len(indexes) > 0:
    for i in indexes.flatten():  #탐지된 객체 있는지 확인
        x, y, w, h = boxes[i]  #객체의 좌표 정보
        print(x, y, w, h)
        label = str(classes[class_ids[i]])  #classes 리스트에서 해당 클래스의 이름 가져오기
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), ((x+w), (y+h)), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0, 255, 0), 2)

    plt.imshow(img)
  
  else:  #탐지된 객체가 없을 때
    print('탐지된 물체가 없습니다.')


# In[17]:


import glob  #glob: 파일 경로들의 리스트를 얻기 위해 사용되는 라이브러리
import random  #random: 난수를 생성하고 리스트에서 무작위로 항목을 선택하는 데 사용되는 라이브러리

paths = glob.glob('testing_images/*.jpg')  #모든 파일 경로들을 저장

img_path = random.choice(paths)  #무작위로 하나 선택

predict_yolo(img_path)  #랜덤 객체로 객체 탐지 수행


# In[18]:


img_path = random.choice(paths)

predict_yolo(img_path)


# # 직접 찍은 영상으로 객체인식

# In[ ]:


import cv2 as cv
import numpy as np

# YOLO 로드
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 가져오기
video = cv.VideoCapture('IMG_4753.mp4')

count = 0
while video.isOpened():
    ret, img = video.read()
    if not ret:
        break
    if int(video.get(1)) % 7 == 0:
        img = cv.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        #이미지 전처리
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 정보를 화면에 표시
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #객체 감지
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    #박스 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y + 30), font, 3, color, 3)

        cv.imshow('object_detection', img)
        key = cv.waitKey(1)
        if key == 32:  # Space
            key = cv.waitKey(0)
        if key == 27:  # ESC
            break
cv.destroyAllWindows()


# # 동영상 프레임 초단위 저장 캡쳐

# In[20]:


import cv2
import os

filepath = 'IMG_4753.mp4'
video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)
    
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("fps :", fps)


# In[ ]:


try:
    if not os.path.exists(filepath[:-4]):
        os.makedirs(filepath[:-4])
except OSError:
    print ('Error: Creating directory. ' +  filepath[:-4])
    
interval = 0.5
count = 0

while(video.isOpened()):
    ret, image = video.read()
    if(int(video.get(1)) % int(fps*interval) == 0): #앞서 불러온 fps 값을 사용하여 0.5초마다 추출
        cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
        print('Saved frame number :', str(int(video.get(1))))
        count += 1
        
video.release()


# # 동영상에서 특정 부분 검출하기

# In[ ]:


#웹캠 이미지 실시간 출력
import cv2
import numpy as np

# 웹 카메라로부터 입력받기 --- (*1)
cap = cv2.VideoCapture(0)
while True:
    # 카메라의 이미지 읽어 들이기 --- (*2)
    _, frame = cap.read()
    # 이미지를 축소해서 출력하기 --- (*3)
    frame = cv2.resize(frame, (500,300))
    # 윈도우에 이미지 출력하기 --- (*4)
    cv2.imshow('OpenCV Web Camera', frame)
    # ESC 또는 Enter 키가 입력되면 반복 종료하기
    k = cv2.waitKey(1) # 1msec 대기
    if k == 27 or k == 13: break

cap.release() # 카메라 해제하기
cv2.destroyAllWindows() # 윈도우 제거하기


# In[ ]:


#노란색 부분을 흰색으로 칠해 이미지를 화면에 출력
import cv2
import numpy as np

# 웹 카메라로부터 입력받기
cap = cv2.VideoCapture(0)
while True:
    # 이미지 추출하고 축소하기
    _, frame = cap.read()
    frame = cv2.resize(frame, (500,300))
    # 색공간을 HSV로 변환하기 --- (*1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    # HSV 분할하기 --- (*2)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    #노란색을 가진 요소만 출력하기 --- (*3)
    img = np.zeros(h.shape, dtype=np.uint8)
    img[((h >= 20) | (h <= 45)) & (s > 50)] = 255
    # 윈도우에 이미지 출력하기 --- (*4)
    cv2.imshow('Yellow Camera', img)
    if cv2.waitKey(1) == 13: break

cap.release() # 카메라 해제하기
cv2.destroyAllWindows() # 윈도우 제거하기


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




