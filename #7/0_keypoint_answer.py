"""
pip install opencv-contrib-python==3.4.2.16
"""
import cv2
import numpy as np
import util
import time

sfit = cv2.xfeatures2d.SIFT_create() # SIFT 검출기 생성
surf = cv2.xfeatures2d.SURF_create() # SURF 검출기 생성

for scale_factor in [0.5, 1.0, 2.0, 10]:
    filepath = 'butterfly.png'  #filepath = 'rectangle.png'
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
    print('>>', gray.shape)

    # SIFT 특징 검출
    sfit_start = time.time()
    kpts = sfit.detect(image=gray, mask=None) # SIFT keypoints 검출
    print("SFIT :", time.time() - sfit_start)

    # SURF 특징 검출
    surf_start = time.time()
    kpts = surf.detect(image=gray, mask=None) # SIFT keypoints 검출
    print("SURF :", time.time() - surf_start)


# keypoint 시각화 (https://docs.opencv.org/2.4/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html#drawkeypoints)
# 기본 버젼
res = cv2.drawKeypoints(image=gray,
                        keypoints=kpts,
                        outImage=None)
# 섬세하게 출력하는 버전
res_with_rich = cv2.drawKeypoints(image=gray,
                                  keypoints=kpts,
                                  outImage=None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 함수 대신 숫자 4를 넣어도 동작함

concatenated = np.hstack((res, res_with_rich))
cv2.imshow('concatenated', concatenated)
cv2.waitKey(0)

cv2.destroyAllWindows()

##############################
# keypoint 결과 출력해보기
##############################
octave_set = set()
layer_set = set()
for i, keypoint in enumerate(kpts):
    octave, layer, scale = util.unpackSIFTOctave(keypoint)
    print('[keypoint #%d] octave: %2d\t layer: %d\t scale: %f' % (i, octave, layer, scale))

    octave_set.add(octave)
    layer_set.add(layer)

print('octave set:', octave_set)
print('layer set:', layer_set)
