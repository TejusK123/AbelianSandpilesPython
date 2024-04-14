import numpy as np
from numba import cuda, numba

import math 

#max holding for n-dimensional sandpile is 2n-1
# cv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='circular', bias=False)
# cv.requires_grad=False
# cv.weight = torch.nn.Parameter(
#     torch.tensor(
#         [[[[ 1., 1., 1.],
#            [ 1., 0., 1.],
#            [ 1., 1., 1.]]]],
#         device=device,
#         dtype=torch.float16
#     ),
#     requires_grad=False,
# )
@cuda.jit
def topple2d(A, B, scale, offset = []):
    
    x, y = cuda.grid(2)
    if x < A.shape[0] and y < A.shape[1]:
        if A[x][y] > 3:
            B[x][y] -= 4 * scale + offset
        if (x+1) < A.shape[0] and A[x+1][y] > 3 * scale + offset:
            B[x][y] += 1 * scale + offset
        if (y+1) < A.shape[1] and A[x][y+1] > 3 * scale + offset:
            B[x][y] += 1 * scale + offset
        if A[x-1][y] > 3 * scale + offset and (x - 1) > -1:
            B[x][y] += 1 * scale + offset
        if A[x][y-1] > 3 * scale and (y - 1) > -1:
            B[x][y] += 1 * scale + offset

        
        
# @cuda.jit
# def topple2d(A, B, laplacian = None):
#     x, y = cuda.grid(2)
#     if laplacian is not None:
#         if x < A.shape[0] and y < A.shape[1]:
#             index = int(x * math.sqrt(laplacian.shape[0]) + y)
#             if A[x][y] >= laplacian[index, index]:
#                 B[x][y] -= laplacian[index, index]
#             if (x+1) < A.shape[0] and A[x+1][y] >= 4:
#                 B[x][y] += 1
#             if (y+1) < A.shape[1] and A[x][y+1] >= 4:
#                 B[x][y] += 1
#             if A[x-1][y] >= 4 and (x - 1) > -1:
#                 B[x][y] += 1
#             if A[x][y-1] >= 4 and (y - 1) > -1:
#                 B[x][y] += 1
	
#     else:
#          if x < A.shape[0] and y < A.shape[1]:
#             if A[x][y] > 3:
#                 B[x][y] -= 4
#             if (x+1) < A.shape[0] and A[x+1][y] > 3:
#                 B[x][y] += 1
#             if (y+1) < A.shape[1] and A[x][y+1] > 3 :
#                 B[x][y] += 1 
#             if A[x-1][y] > 3 and (x - 1) > -1:
#                 B[x][y] += 1 
#             if A[x][y-1] > 3 and (y - 1) > -1:
#                 B[x][y] += 1
         

@cuda.jit
def topple3d(A,B):
    x,y,z = cuda.grid(3)
#index is within bounds
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

        
     
def assertabelian(A):
    check = A.copy()
    one = np.all([check[i][i] > 0 for i in range(A.shape[0])])
    two = np.all([np.sum(check[i]) >= 0 for i in range(A.shape[0])])
    three = np.sum(check) > 0
    four = np.linalg.slogdet(check)[0] != 0
    np.fill_diagonal(check, 0)
    five = np.all(check <= 0)
    
    return([one, two, three, four, five])
        

from PIL import Image			
def create_visualization(a, name):
        a = a.astype('uint8')
        
        img = Image.fromarray(a, 'P')
        arr = np.array([255,255,255,0,0,204,102,0,102,0,0,0,255,0,0])  #0 = White, 1 = blue, 2 = purple, 3 = black
        
        img.putpalette(arr.tolist(), rawmode = 'RGB')

        img.save(name)


def stabilize(A, data = False, scale = 1, visualize = False, ref = None, offset = 0, laplacian = None):
# A is the input array
# B is the output array
# data is a boolean that determines whether to return the data
# or just the number of iterations
# returns the number of iterations it took to stabilize the array

    
# determine if input array is 2d or 3d
    if len(A.shape) == 2:
        threadsperblock = (32,32)
        blockspergridx = int(math.ceil(A.shape[0] / threadsperblock[0]))
        blockspergridy = int(math.ceil(A.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergridx, blockspergridy)
        topple = topple2d
        n = 3 * scale

    if len(A.shape) == 3:
        threadsperblock = (16,16,4)
        blockspergridx = int(math.ceil(A.shape[0] / threadsperblock[0]))
        blockspergridy = int(math.ceil(A.shape[1] / threadsperblock[1]))
        blockspergridz = int(math.ceil(A.shape[2] / threadsperblock[2]))
        blockspergrid = (blockspergridx, blockspergridy, blockspergridz)
        topple = topple3d
        n = 5 * scale
        

    B = A.copy()
    Flag = True 
    count = 0
    outputdata = B.copy()
    similarity_list = []
    a_G = cuda.to_device(A)
    b_G = cuda.to_device(B)
    while Flag:
        count += 1
        
        # l_G = cuda.to_device(laplacian)
        top = np.max(a_G)
        if top <= n:
            Flag = False 
        else:
            topple[blockspergrid, threadsperblock](a_G, b_G, scale, offset)
            
            a_G = b_G 
            b_G = a_G 
        # l_G.copy_to_host(laplacian)
        if data:
            outputdata = np.dstack((outputdata, B))
             
        if visualize:
            create_visualization(B, f"top{count}.png")
        if ref is not None:
            similarity_list.append(np.sum(B == ref)/B.shape[0] ** 2)
    a_G.copy_to_host(A)
    b_G.copy_to_host(B)   
    if ref is not None:
        return B, similarity_list
    else:
        return B, outputdata
   



if __name__ == "__main__":
    
    import time 

    start_time = time.time()

    A = np.full((100,100),6)
    z = stabilize(A)
    create_visualization(z[0], "topple.png")
    print("--- %s seconds ---" % (time.time() - start_time))
              
   





			
		

		
		
	
	
			
