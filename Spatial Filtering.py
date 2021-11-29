import cv2
import math as m
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("./inputimage.jpg",0)
t_img=cv2.imread("./inputimage.jpg",0)
#----------------------------------------------Q3-----------------------------------------------
def histogram_equalization(img):
	cv2.imshow("input-image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
	

	h=normalized_hist(img)
	plt.title('Original Histogram')
	plt.plot(h)
	plt.show()
	

	cdf=np.array(np.cumsum(h))
	tk=np.uint8(255*cdf)
	M,N=img.shape
	Z=np.zeros_like(img)

	for i in range(M):
		for j in range(N):
			Z[i,j]=tk[img[i,j]]
	H=normalized_hist(Z)
	plt.plot(H)
	plt.title('Equalized Histogram')
	plt.show()

	cv2.imshow("equalized-image",Z)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
def normalized_hist(img):
	m, n = img.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[img[i, j]]+=1
	return np.array(h)/(m*n)

#histogram_equalization(img)

#----------------------------------------------Q4-----------------------------------------------
def histogram_matching(img,t_img):
	cv2.imshow("input-image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

	cv2.imshow("target-image",t_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	gamma=0.5
	gamma_transformed=np.array(255*(img/255)**gamma,dtype='uint8')
	cv2.imshow("gamma_transformed",gamma_transformed)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	t_gamma_transformed=np.array(255*(t_img/255)**gamma,dtype='uint8')
	cv2.imshow("gamma_transformed of target-image",t_gamma_transformed)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	H=normalized_hist(img)
	plt.plot(H)
	plt.title('Histogram of Input Image')
	plt.show()

	t_H=normalized_hist(t_img)
	plt.plot(H)
	plt.title('Histogram of target Image')
	plt.show()

	G=normalized_hist(gamma_transformed)
	plt.plot(G)
	plt.title('Histogram of Gamma Transformed Image')
	plt.show()

	G=normalized_hist(t_gamma_transformed)
	plt.plot(G)
	plt.title('Histogram of Gamma Transformed Target Image')
	plt.show()

#histogram_matching(img,t_img)

#----------------------------------------------Q5-----------------------------------------------
def conv_trans(img):
	img_c=img.copy()
	for i in range(3):
		for j in range(3):
			img_c[i][j]=img[3-i-1][3-j-1]
	print("Rotated Filter")
	print(img_c)
	return img_c
def conv(img,kernel):
	kernel=conv_trans(kernel)
	img=padding(img)

	img_h=5
	img_w=5

	kernel_h=3
	kernel_w=3

	h=1
	w=1

	image_conv=np.zeros((5,5),dtype='uint8')
	for i in range(h,img_h-1):
		for j in range(w,img_w-1):
			sum=0
			for m in range(3):
				for n in range(3):
					sum=sum+kernel[m][n]*img[i-1+m][j-1+n]

			image_conv[i][j]=sum
	print("Convoluted Matrix")
	return image_conv
def padding(img):
	img=np.insert(img,0,[0],axis=0)
	img=np.insert(img,3,[0],axis=0)

	img=np.insert(img,0,0,axis=1)
	img=np.insert(img,3,0,axis=1)
	return img
	# print(img)

img=np.array([[0,0,0] , [0,1,0] , [0,0,0]])
filt=np.array([[1,2,3] , [4,5,6] , [7,8,9]])
# img=np.random.randint(256,size=(3,3))
# filt=np.random.randint(256,size=(3,3))
# print("Input matrix")
# print(img)
# print("Filter matrix")
# print(filt)
# print(conv(img,filt))