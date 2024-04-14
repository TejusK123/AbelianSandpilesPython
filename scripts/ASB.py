import numpy as np
from numba import cuda, numba
import math
from libpysal.weights import lat2W
from PIL import Image
import PIL
import os
from random import randint
import cv2
import matplotlib.pyplot as plt
import csv
import math
from sklearn.preprocessing import MinMaxScaler
import re
'''
wants:
-incorporate different shaped sandpiles i.e circular, triangular
-figure out surface method of calculating identity
-learn Javascript to make a website
-add an estimated time bar for creating identities
-add a randomely drop sand option
'''
a = 20
#possily can combine these two topple functions
@cuda.jit
def topple_tri(A,B):
	x, y = cuda.grid(2)

	if x < A.shape[0] and y < A.shape[1] and x <= y:

		# if A[x][y] > 3:
		# 	B[x][y] -= 4
		# if A[x+1][y] > 3 and (x+1) < A.shape[0]:
		# 	B[x][y] += 1
		# if A[x][y+1] > 3 and (y+1) < A.shape[1]:
		# 	B[x][y] += 1
		# if A[x-1][y] > 3 and (x - 1) > -1:
		# 	B[x][y] += 1
		# if A[x][y-1] > 3 and (y - 1) > -1:
		# 	B[x][y] += 1

		if x == 0 and y == 0:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y+1] > 3:
				B[x][y] += 1
			if A[x+1][y] > 3:
				B[x][y] += 1
		elif x == A.shape[0] - 1 and y == A.shape[1] - 1:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y-1] > 3:
				B[x][y] += 1
			if A[x-1][y] > 3:
				B[x][y] += 1
		elif x == 0 and y == A.shape[1] - 1:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y-1] > 3:
				B[x][y] += 1
			if A[x+1][y] > 3:
				B[x][y] += 1
		elif x == A.shape[0] - 1 and y == 0:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y+1] > 3:
				B[x][y] += 1
			if A[x-1][y] > 3:
				B[x][y] += 1
		elif x == 0:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y+1] > 3:
				B[x][y] += 1
			if A[x+1][y] > 3:
				B[x][y] += 1
			if A[x][y-1] > 3:
				B[x][y] += 1

		elif y == 0:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y+1] > 3:
				B[x][y] += 1
			if A[x+1][y] > 3:
				B[x][y] += 1
			if A[x-1][y] > 3:
				B[x][y] += 1

		elif x == A.shape[0] - 1:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y+1] > 3:
				B[x][y] += 1
			if A[x-1][y] > 3:
				B[x][y] += 1
			if A[x][y-1] > 3:
				B[x][y] += 1

		elif y == A.shape[1] - 1:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x-1][y] > 3:
				B[x][y] += 1
			if A[x+1][y] > 3:
				B[x][y] += 1
			if A[x][y-1] > 3:
				B[x][y] += 1
		
		else:
			if A[x][y] > 3:
				B[x][y] -= 4
			if A[x][y+1] > 3:
				B[x][y] += 1
			if A[x+1][y] > 3:
				B[x][y] += 1
			if A[x][y-1] > 3:
				B[x][y] += 1
			if A[x-1][y] > 3:
				B[x][y] += 1

@cuda.jit
def topplecustom2d(A,B):

	x, y = cuda.grid(2)

	if x < A.shape[0] and y < A.shape[1]:
		if A[x][y] > 3:
			B[x][y] -= 4
			if A[x+1][y] == -1:
				B[x][y] += 1
			if A[x-1][y] == -1:
				B[x][y] += 1
			if A[x][y+1] == -1:
				B[x][y] += 1
			if A[x][y-1] == -1:
				B[x][y] += 1
		if (x+1) < A.shape[0] and A[x+1][y] > 3 and A[x][y] != -1:
			B[x][y] += 1
		if (y+1) < A.shape[1] and A[x][y+1] > 3 and A[x][y] != -1:
			B[x][y] += 1
		if A[x-1][y] > 3 and (x - 1) > -1 and A[x][y] != -1:
			B[x][y] += 1
		if A[x][y-1] > 3 and (y - 1) > -1 and A[x][y] != -1:
			B[x][y] += 1


@cuda.jit
def topple3d(A,B):
	x,y,z = cuda.grid(3)

	if x < A.shape[0] and y < A.shape[1] and z < A.shape[2]:
		if A[x][y][z] > 5:
			B[x][y][z] -= 6
		if (x+1) < A.shape[0] and A[x+1][y][z] > 5:
			B[x][y][z] += 1
		if (y+1) < A.shape[1] and A[x][y+1][z] > 5:
			B[x][y][z] += 1
		if (z+1) < A.shape[2] and A[x][y][z+1] > 5:
			B[x][y][z] += 1
		if A[x-1][y][z] > 5 and (x-1) > -1:
			B[x][y][z] += 1
		if A[x][y-1][z] > 5 and (y-1) > -1:
			B[x][y][z] += 1
		if A[x][y][z-1] > 5 and (z-1) > -1:
			B[x][y][z] += 1
@cuda.jit
def topple1d(A,B):
	x = cuda.grid(1)

	if x < A.shape[0]:
		if A[x] > 1:
			B[x] -= 2
		if (x+1) < A.shape[0] and A[x+1] > 1:
			B[x] += 1
		if A[x-1] > 1 and (x-1) > -1:
			B[x] += 1
@cuda.jit
def topple2d(A, B):

	


	x, y = cuda.grid(2)
	# if Shape == 'tri':
	# 	tricond = (x <= y)
	# else: 
	# 	tricond = None

	if x < A.shape[0] and y < A.shape[1]:
	# 	if A[x][y] > 3:
	# 		B[x][y] -= 4
	# 	if (x+1) < A.shape[0] and A[x+1][y] > 3:
	# 		B[x][y] += 1
	# 	if (y+1) < A.shape[1] and A[x][y+1] > 3:
	# 		B[x][y] += 1
	# 	if A[x-1][y] > 3 and (x - 1) > -1:
	# 		B[x][y] += 1
	# 	if A[x][y-1] > 3 and (y - 1) > -1:
	# 		B[x][y] += 1

		#case by case long method faster than above method
		#for n-dimensions, top is only feasible
		# if A[x][y] > 3:
		# 	B[x][y] -= 4
		# 	if A[x+1][y] == -1:
		# 		B[x][y] += 1
		# 	if A[x-1][y] == -1:
		# 		B[x][y] += 1
		# 	if A[x][y+1] == -1:
		# 		B[x][y] += 1
		# 	if A[x][y-1] == -1:
		# 		B[x][y] += 1
		# if (x+1) < A.shape[0] and A[x+1][y] > 3 and A[x][y] != -1:
		# 	B[x][y] += 1
		# if (y+1) < A.shape[1] and A[x][y+1] > 3 and A[x][y] != -1:
		# 	B[x][y] += 1
		# if A[x-1][y] > 3 and (x - 1) > -1 and A[x][y] != -1:
		# 	B[x][y] += 1
		# if A[x][y-1] > 3 and (y - 1) > -1 and A[x][y] != -1:
		# 	B[x][y] += 1
		

		if A[x][y] > 3 :
			B[x][y] -= 4
			if A[x+1][y] == -1 :
				B[x][y] += 1 
			if A[x-1][y] == -1:
				B[x][y] += 1
			if A[x][y+1] == -1:
				B[x][y] += 1
			if A[x][y-1] == -1:
				B[x][y] += 1
		if (x+1) < A.shape[0] and A[x+1][y] > 3:
			B[x][y] += 1
		if (y+1) < A.shape[1] and A[x][y+1] > 3:
			B[x][y] += 1
		if A[x-1][y] > 3 and (x - 1) > -1:
			B[x][y] += 1
		if A[x][y-1] > 3 and (y - 1) > -1:
			B[x][y] += 1


	



#must have a result matrix because each of the threads in the gpu work simultaneously
#results in incorrect output if just editing original matrix

def stabilizesq(A, data = False):
	B = np.copy(A)
	#possibly not most efficient threads per block
	#also need to figure out how to stop the host from repeatedly copying to and from device

	threadsperblock = (32,32)
	blockspergridx = int(math.ceil(A.shape[0] / threadsperblock[0]))
	blockspergridy = int(math.ceil(A.shape[1] / threadsperblock[1]))


	blockspergrid = (blockspergridx, blockspergridy)

	Flag = True
	
	while Flag:
		
		A_G = cuda.to_device(A)
		B_G = cuda.to_device(B)
		top = np.max(A_G)

		if top <= 3:
			Flag = False
		else:

			topple2d[blockspergrid, threadsperblock](A_G,B_G)
			A_G = B_G
			B_G = A_G
			A_G.copy_to_host(A)
			B_G.copy_to_host(B)
			# if data == True:
			# 	alls.append(B.flatten())

				

	
	return(B)
def stabilize3drec(A,B):
	threadsperblock = (16,16,4)
	blockspergridx = int(math.ceil(A.shape[0] / threadsperblock[0]))
	blockspergridy = int(math.ceil(A.shape[1] / threadsperblock[1]))
	blockspergridz = int(math.ceil(A.shape[2] / threadsperblock[2]))
	blockspergrid = (blockspergridx, blockspergridy, blockspergridz)
	Flag = True
	while Flag:


		
		a_G = cuda.to_device(A)
		b_G = cuda.to_device(B)
		top = np.max(a_G)
		

		if top <= 5:
			Flag = False
		else:
			topple3d[blockspergrid, threadsperblock](a_G, b_G)
			a_G = b_G
			b_G = a_G

		a_G.copy_to_host(A)
		b_G.copy_to_host(B)
			

	return(B)


def stabilizetr(A,B):
	
	threadsperblock = (32,32)
	blockspergridx = int(math.ceil(A.shape[0] / threadsperblock[0]))
	blockspergridy = int(math.ceil(A.shape[1] / threadsperblock[1]))


	blockspergrid = (blockspergridx, blockspergridy)

	Flag = True
	
	while Flag:
		a_G = cuda.to_device(A)
		b_G = cuda.to_device(B)
		top = np.max(a_G)

		if top <= 3:
			Flag = False
		else:
			topple_tri[blockspergrid, threadsperblock](a_G,b_G)
			a_G = b_G
			b_G = a_G
			a_G.copy_to_host(A)
			b_G.copy_to_host(B)

	return(B)


#formula I = stabilize(2u - stabilize(2u))
#u = max stable configuration

def identity_formula_3drec(x,y,z):
	u = np.full((x,y,z), 10)
	r_u = u[:]
	p = u.copy()

	z = stabilize3drec(u, r_u)
	print("stab(2u) calculated")

	

	m = p - z

	r_m = m.copy()

	ans = stabilize3drec(m,r_m)

	return(ans)




def identity_formula_rec(x,y):
	u = np.full((x, y), 6)
	r_u = u[:]
	

	z = stabilizesq(u, r_u)
	print("stab(2u) calculated")
	
	p = np.full((x,y), 6)

	m = p-z

	r_m = m.copy()

	ans = stabilizesq(m,r_m)


	return(ans)

#create new topple alg

def identity_formula_tri(x,y):
	print("starting now")
	u = np.full((x,y), 6)
	for i in range(x):
		for j in range(y):
			if j < i:
				u[i][j] = -1
	

	r_u = u[:]

	z = stabilizetr(u, r_u)

	print("stab(2u) calculated")

	p = np.full((x,y), 6)
	for q in range(x):
		for r in range(y):
			if r < q:
				p[q][r] = -1

	m = p - z
	for q in range(x):
		for r in range(y):
			if r < q:
				m[q][r] = -1
	

	r_m = m.copy()

	ans = stabilizetr(m, r_m)
	for j in range(x):
		for k in range(y):
			if k < j:
				ans[j][k] = 4

	return(ans)



	

#get the estimation curve for k
def identity_function(x,y):
	xx, yy = np.meshgrid()
def identity_iterative_rec(x,y, vid):


	u = np.zeros((x,y))

	for i in range(x):
		for j in range(y):
			if i == 0 or j == 0 or i == x - 1 or j == y - 1:
				u[i][j] = 1


	u[0][y-1] = 2
	u[0][0] = 2
	u[x-1][y-1] = 2
	u[x-1][0] = 2

	u_r = u[:]
	cur_img = 0
	dir_num = 2
	if vid == True:
		dir_num = str(dir_num)

		name1 = f"img{dir_num}"

		#will need alternate solution

		dir_num = int(dir_num)
		dir_num += 1

		directory = name1
		parent_dir = "C:/Users/tkoti/Desktop/Python_learning/python_work/Math/Sandpiles"
		path = os.path.join(parent_dir, directory)	
		os.mkdir(path)


		cur_img = str(cur_img)
		name = f"{parent_dir}/{directory}/{cur_img}.png"
		print(name)
		create_visualization(u,name)
		cur_img = int(cur_img)
		cur_img += 1

	check = stabilizesq(2*u, 2*u_r)

	while (check == u).all() == False:
		if vid == True:
			

			cur_img = str(cur_img)
			name = f"{parent_dir}/{directory}/{cur_img}.png"
			print(name)
			create_visualization(check, name)
			cur_img = int(cur_img)
			cur_img += 1
		u = check
		u_r = u[:]
		check = stabilizesq(2*u, 2*u_r) 

	
	ans = check



	#lastly use openCV to generate a video
	return(check)

def square_laplacian(n):
	w = lat2W(n,n)

	a = np.zeros((n*n, n*n), dtype = 'uint8')

	np.fill_diagonal(a, 4)

	L = a - (w.full()[0])

	return(L)

#format saved file name as 'example.{whatever extension you want}'

#figure out how to create a visualization for 3d sandpile
#in the future also create a 3d projection of a 4d sandpile
def create_visualization(a, name):
	a = a.astype('uint8')
	
	img = Image.fromarray(a, 'P')
	arr = np.array([255,255,255,0,0,204,102,0,102,0,0,0,52,52,52])  #0 = White, 1 = blue, 2 = purple, 3 = black
	# arr = list(np.array([240,240,240,228,228,228,216,216,216,204,204,204,192,192,192,180,180,180,168,168,168,156,156,156,144,144,144,132,132,132,120,120,120,108,108,108,96,96,96,84,84,84,72,72,72,60,60,60,48,48,48,36,36,36,24,24,24,12,12,12,0,0,0]))
	
	# arr = reversed(arr)
	img.putpalette(arr, rawmode = 'RGB')

	img.save(name)

	# plt.imshow(a, interpolation='none', cmap='gray')
	# plt.savefig(name)

#figure out how to change directory automatically
def generate_video():
    image_folder = 'C:/Users/tkoti/Desktop/Python_learning/python_work/Math/Sandpiles/img1' # make sure to use your folder
    video_name = 'mygeneratedvideo.avi'
    os.chdir("C:/Users/tkoti/Desktop/Python_learning/python_work/Math/Sandpiles/img1")
      
    images = [img for img in os.listdir(image_folder)]
     
    # Array images should only consider
    # the image files ignoring others if any
    images.sort(key = len)
    print(images) 
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release() 

# generate_video()


# print(identity_iterative_rec(50,50, True))
# create_visualization(identity_formula_rec(2000,2000), '2000x2000_identity.png')
# create_visualization(identity_rectangular(3,3), 'test.png')
# test1 = identity_formula_tri(2000,2000)

#76.8 500x500 tri
#178.5 500x500 sq
# test2 = identity_formula_rec(500,500)
# create_visualization(test1, '2000x2000tri.png')

#only square matrices currently


#formula I = stabilize(2u - stabilize(2u))


def add_random(A, n):

	x = A.shape[0]
	y = A.shape[1]

	for itera in range(n):
		randx = randint(0,x - 1)
		randy = randint(0,y - 1)
		A[randx][randy] += 1
		A = (stabilizesq(A))
		create_visualization(A, f"C:/Users/tkoti/Desktop/Programming/misc/Math/Sandpiles/imgdump/imgdump{itera}.png")
	
	# print("stabilizing now")
	

	return(A)

if __name__ == "__main__":

	# test = stabilizesq(np.random.randint(1,4,size = (10,10)) + np.full((10,10), 3))
	# I = identity_formula_rec(10,10)

	# testrecurrence = stabilizesq(test + I)

	# if (testrecurrence == test).all():
	# 	print("recurrent")
	# else:
	# 	print("nonrecurrent")

	# mask = square_laplacian(10)



	# print(mask)

	# test1 = test + mask
	# create_visualization(test, 'preperturbation.png')
	# create_visualization(stabilizesq(test1), 'postperturbation.png')

	# z = (test == test1)
	# count = 0
	# for i in range(10):
	# 	for j in range(10):
	# 		if z[i][j] == False:
	# 			count += 1
	# 			print(i,j)

	# print(count)



	testimage = np.array(PIL.Image.open(r"C:\Users\tkoti\Desktop\Programming\misc\Math\Sandpiles\testimages\monalisa.png"))
	testimage = testimage[:125, :128]
	# import matplotlib.pyplot as plt


	testimage = testimage @ [0.2126, 0.7152, 0.0722]
	testimage = testimage[:,:]
	print(testimage.shape)
	#plt.imshow(testimage, cmap = 'gray')
	#plt.show()



	# print(testimage.shape)
	testimage = ((testimage)).astype('uint8')

	print(max(testimage.flatten()), min(testimage.flatten()))
	# create_visualization(testimage, 'iiiiiii.png')



	# plt.imshow(testimage)
	# plt.show()
	mask = square_laplacian(125)
	# n = np.random.randint(0,2, (1,125*125))


	# where_add = ((n @ mask)).reshape(125,125)

	# perturbed = testimage + where_add

	# plt.imshow(perturbed)
	# plt.show()

	# relaxed = stabilizesq(perturbed)

	# plt.imshow(perturbed)
	# plt.show()

	testimage = stabilizesq(identity_formula_rec(125,125) + np.full((125,125), 6))
	blank = np.full((128,128), 0)

	N = np.array([0 for i in range(125**2)])
	N = N.reshape(125,125)
	N[3, 15:20] = 1
	N[4, 15:20] = 1
	N[5, 15:20] = 1
	N[6, 15:20] = 1
	N[7, 15:20] = 1
	N[8, 15:20] = 1

	N[21, 200:254] = 1
	N[22, 200:254] = 1
	N[23, 200:254] = 1
	N[24, 200:254] = 1
	N[25, 200:254] = 1
	N[26, 200:254] = 1

	N[50, 50:100] = 1
	N[51, 50:100] = 1
	N[52, 50:100] = 1
	N[53, 50:100] = 1
	N[54, 50:100] = 1

	N = N.reshape(1,125*125)


	newmask = (N @ mask).reshape(125,125) 

	testimage1 = stabilizesq(testimage + newmask)
	count = 0
	create_visualization(testimage1, 'testimageperturbed.png')
	while not (np.all(testimage1 == stabilizesq(testimage1 + newmask))):

		count += 1
		print(count)
		

		testimage1 = testimage1 + newmask

		testimage1 = stabilizesq(testimage1)

		create_visualization(testimage1, f'testimageperturbed{count}.png')




	# print(np.all(testlayer1 == testlayer))


	#use ratios i.e xvalues/len x dimension


	# data_folder = 'C:/Users/tkoti/Desktop/Python_learning/python_work/Math/Sandpiles/stoichastic_sinks'
	# images = [img for img in os.listdir(data_folder) if img.endswith(".png")]
	# test = []
	# for item in images:
	# 	print(item)
	# 	path = "C:/Users/tkoti/Desktop/Python_learning/python_work/Math/Sandpiles/stoichastic_sinks/" + item
	# 	img = np.array(PIL.Image.open(path))
	# 	img += 1
	# 	img = np.pad(img, 1)
	# 	numsinks = item[18:]
	# 	numsinks = int(numsinks.strip("sinks.png"))
		
	# 	for i in range(501):
	# 		for j in range(501):
	# 			if img[i][j] == 5:
	# 				x = (i + 1)/500
	# 				y = (j + 1)/500
	# 				p1 = img[i-1][j-1]
	# 				p2 = img[i-1][j]
	# 				p3 = img[i-1][j+1]
	# 				p4 = img[i][j-1]
	# 				p5 = img[i][j+1]
	# 				p6 = img[i+1][j-1]
	# 				p7 = img[i+1][j]
	# 				p8 = img[i+1][j+1]

	# 				test.append([numsinks,x,y,p1,p2,p3,p4,p5,p6,p7,p8])
					
	# print(test)

	# # for i in range(len(test)):
	# # 	for j in range(len(test[i])):
	# # 		if int(test[i][j]) == test[i][j] and test[i][j] != 0 and test[i][j] <= 5:
	# # 			test[i][j] -= 1
	# # print("\n")
	# print(test)

	# header = ["numsinks", "xpercent", "ypercent", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]

	# with open("data500x500.csv", "w", encoding = "UTF8", newline = '') as f:
	# 	writer = csv.writer(f)

	# 	writer.writerow(header)
	# 	writer.writerows(test)



	# testimg = np.array(PIL.Image.open(path))



	# testimg += 1

	# testimg = np.pad(testimg, 1)

	# for i in range(2):
	# 	for j in range(501):
	# 		if testimg[i][j] == 5:
	# 			x = (i + 1)/500
	# 			y = (j + 1)/500
	# 			p1 = testimg[i-1][j-1]
	# 			p2 = testimg[i-1][j]
	# 			p3 = testimg[i-1][j+1]
	# 			p4 = testimg[i][j-1]
	# 			p5 = testimg[i][j+1]
	# 			p6 = testimg[i+1][j-1]
	# 			p7 = testimg[i+1][j]
	# 			p8 = testimg[i+1][j+1]

	# 			test.append([x,y,p1,p2,p3,p4,p5,p6,p7,p8])


	# print(test)
	# print(testimg[0:3, 430:433])



	# sinks = [(i,j) for i in range(499) for j in range(499) if testimg[i, j] == 5]

	# print(sinks)
	#if recurrent, stab(I + matrix) will = matrix
	#if not it wont
	#first test stoichastic sinks 
	#started with full 6 then added random sinks
	# for number in range(0,1000,25):

	# 	l = 500
	# 	w = 500
	# 	name = f"stoichastic_sinks/3x3/testrandom500x500_{number}3x3sinks.png"
	# 	custom = np.full((l,w), 6)
	# 	n = number
	# 	print("starting")
	# 	for z in range(n):
	# 		x = randint(0,l - 1)
	# 		y = randint(0,w - 1)

	# 		custom[x:x+2, y:y+2] = -1

	# 	customr = custom[:]

	# 	print("toppling custom")

	# 	stabilizesq(custom, customr)


	# 	customr = stabilizesq(custom, customr)
	# 	for i in range(l - 1):
	# 		for j in range(w - 1):
	# 			if customr[i][j] == -1:
					
	# 				customr[i][j] = 4


	# 	create_visualization(customr, name)

	# name = f"{parent_dir}/{directory}/{cur_img}.png"
	# print("toppling square")
	# reference = np.full((500,500), 6)
	# referencer = reference[:]

	# create_visualization(stabilizesq(reference, referencer), 'testrandomref.png')
	# test1c = test1 + Ident


	# referencetri = np.full((200,200), 6)
	# for i in range(200):
	# 		for j in range(200):
	# 			if j < i:
	# 				referencetri[i][j] = -1

	# referencetrir = referencetri[:]

	# referencetrir = stabilizetr(referencetri,referencetrir)
	# for i in range(200):
	# 	for j in range(200):
	# 		if referencetrir[i][j] == -1:
				
	# 			referencetrir[i][j] = 4


	# create_visualization(referencetrir, 'testrandomreftri.png')

	# test1cr = test1c[:]

	# stabilizesq(test1c, test1cr)

	# test2c = test2 + Ident

	# test2cr = test2c[:]

	# stabilizesq(test2c, test2cr)

	# print(test2cr)
	# print(test2c)
	# print(test1cr)
	# print(test1c)

	# start = np.full((3,3), 0)

	# for i in range(10):
	# 	start = add_random(start, 1)
	# 	print(i)
	# 	full = start + Ident
	# 	fullr = full[:]
	# 	stabilizesq(full, fullr)
	# 	if (start == fullr).all():
	# 		allsr.append(fullr)
	# 	else:
	# 		alls.append(fullr)

	# print(alls)
	# if alls != []:
	# 	print("nonrecurrent some")
	# 	print(alls[0])


	# 	test1 = alls[0] + Ident

	# 	test1r = test1[:]


	# 	stabilizesq(test1, test1r)


	# 	print(test1r)


	# print(allsr)
	# print(allsr[0])
	# test2 = allsr[0] + Ident

	# test2r = test2[:]

	# stabilizesq(test2, test2r)

	# print(test2r)




	# alls = (np.concatenate([item for item in alls])).tolist()


	# alls = [int(item) for item in alls]
	# alls.append("end")

	# print(alls)
	# alls = "".join(str(item) for item in alls)
	# identity_iterative_rec(1000,500, True)
	# test3d = np.zeros((3,3,3))
	# print(test3d)

	# print(test3d)
	#178sec, 178.2 new top
	#171, 173, sec old top
	# create_visualization(identity_formula_rec(200,300), 'testnewtoople.png')
	# identity_iterative_rec(500,1000, True)


	# z = (identity_formula_3drec(50,50,50))
	# print(z)

	# with open("55x55recurornot.txt", 'w') as f:
	# 	f.write(alls)
	# 	f.close()











