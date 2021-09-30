import cv2
import numpy as np

#모델영상
img_m = cv2.imread('model.jpg')
hsv_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
h1 = img_m.shape[0]
w1 = img_m.shape[1]
hist_m = cv2.calcHist([hsv_m], [0,1], None, [180, 256], [0,180,0,256])

#입럭영상
img_i = cv2.imread('hand.jpg')
hsv_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)
h2 = img_i.shape[0]
w2 = img_i.shape[1]
hist_i = cv2.calcHist([hsv_i], [0,1], None, [180, 256], [0,180,0,256])

#히스토그램 정규화
hist_m = hist_m/(h1*w1) #h1*w1 대신 img_m.size 로도 접근가능
hist_i = hist_i/(h2*w2)
print("max of hist_m : %f" % hist_m.max())
print("max of hist_i : %f" % hist_i.max())

#비율 히스토그램 계산
hist_r = hist_m/(hist_i+1e-7)
hist_r = np.minimum(hist_r, 1.0)
print("range of hist_r: [%.1f, %.1f]" % (hist_r.min(), hist_r.max()))


#히스토그램 역투영 수행
result = np.zeros_like(img_i, dtype='float32') #입력 영상과 동일 크기의 배열 생성(0으로 초기화)
h, s, v = cv2.split(hsv_i)  #채널 분리
for i in range(h2):
    for j in range(w2):     #모든 픽셀 처리
        h_value = h[i, j]   #(i ,j)의 hue값
        s_value = s[i, j]   #(i, j)의 saturation값
        result[i, j] = hist_r[h_value, s_value] #신뢰도 점수를 결과 이미지에 저장

#이진화 수행 : 화소 값이 0.02 임계값보다 크면 255, 아니면 0
ret, threshold = cv2.threshold(result, 0.02, 255, cv2.THRESH_BINARY)
cv2.imwrite('result.jpg', threshold)

#모폴로지 연산 적용
kernel = np.ones((13,13))
improved = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
#improved = cv2.erode(ret,None)
cv2.imwrite('mophology.jpg', improved)