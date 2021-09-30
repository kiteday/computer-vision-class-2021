import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

plt.figure(figsize=(17, 5)) #전체 피규어 사이즈 결정

img = cv2.imread('hand.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height = img.shape[0]
width = img.shape[1]
plt.subplot(1, 3, 1) #피규어를 1행 3열로 나누고 첫번째 영역에 그림
plt.imshow(img)
plt.axis('off')
plt.title('original')


# OpenCV를 이용한 변환 행렬 도출
center = (width / 2, height / 2)
cv_M = cv2.getRotationMatrix2D(center, 90, 1.0)  # 회전 방향이 반시계방향(CCW; Counter Clock-Wise)
#중심으로 90도 회전 결과
cv_result = cv2.warpAffine(img, cv_M, (width, height))
print('>> OpenCV Rotation matrix')
print(cv_M, end='\n\n')

plt.subplot(1, 3, 2)
plt.imshow(cv_result)
plt.axis('off')
plt.title('cv_result')

# 직접 도출한 행렬을 이용한 회전 변환

fwd_tran_M = np.array([[1, 0, -center[0]],
                       [0, 1, -center[1]],
                       [0, 0,  1]]) #이미지의 기준점을 원점으로 이동
rot_M = np.array([ [ 0, 1, 0],
                   [-1, 0, 0],
                   [ 0, 0, 1]]) #반시계 회전
bwd_tran_M = np.array([[1, 0, center[0]],
                       [0, 1, center[1]],
                       [0, 0,  1]]) #다시 이동했던만큼 복귀

tmp = np.matmul(rot_M, fwd_tran_M)
my_M = np.matmul(bwd_tran_M, tmp)
my_M = my_M[0:2, 0:3]
print('>> My matrix')
print(my_M)
my_result = cv2.warpAffine(img, my_M, (width, height))
#cv2.warpAffine함수는 3,3이 아니라 2,3을 넘겨줘야함
plt.subplot(1, 3, 3)
plt.imshow(my_result)
plt.axis('off')
plt.title('my_result')

# figure 출력
plt.tight_layout()
plt.show()

#전방 기하변한은 엘리어싱 문제가 발생하므로 후방기하연산을 진행해야함 (openCV는 후방)
#Bilinear interpolation 사용 필요