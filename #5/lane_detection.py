import cv2
import pipeline

img = cv2.imread('./test_images/solidWhiteRight.jpg')

result = pipeline.run(img)

cv2.imshow('result', result)
cv2.waitKey(0)
#cv2.imwrite('resut.png',result) #결과를 저장하는 것
cv2.destroyAllWindows() #창을 끔

# # 동영상 테스트
# cap = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')
#
# while True:   #매 프레임을 읽기
#     ok, frame = cap.read()
#     if not ok:
#         break
#
#     result = pipeline.run(frame)
#
#     cv2.imshow('result', result)
#     key = cv2.waitKey(1)  # -1
#     if key == ord('x'):
#         break #x키를 누르면 종료
#
# cap.release()
# cv2.destroyAllWindows()

