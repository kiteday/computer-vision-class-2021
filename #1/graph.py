import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image', img) #이미지 띠우기
# cv2.waitKey() #키를 누르기 전까지 창이 닫히지 않게 함

histo=cv2.calcHist([img],[0],None,[256],[0,256])

# height = img.shape[0]
# width = img.shape[1]
#
# Histo = []
# AHisto = []
# for i in range(0,height):
#     for j in range(0,width):
#         a=img[i,j]
#         Histo.append(a)
#
# for i in range(0, 255): #인덱스를 위한 초기화
#     AHisto.append(0)
#
# for i in range(0, 255):
#     for j in range(0,i):
#         num = Histo[j]
#         sum = AHisto[i]
#         AHisto[i] = sum + num

#plt.hist(Histo)
plt.plot(histo)
plt.show()

# for i in range(0,255):
#     for j in range(0, i):
#         #AHisto[i] += Histo[j]
#         AHisto[i].append(Histo[j])



# cv2.imshow('image', img)
# cv2.waitKey()

#print(AHisto)


a = [1,2,3,1,2,1]

plt.plot(a)
plt.show()
