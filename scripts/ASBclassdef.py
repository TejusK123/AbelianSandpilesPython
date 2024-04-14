from libpysal.weights import lat2W
from topple import stabilize, create_visualization
import numpy as np
from PIL import Image
import tkinter as tk
import matplotlib.pyplot as plt
import time 

"""
ToDO:
Custom toppling rules
1dimensional toppling


"""



class Sandpile:

    def __init__(self):
        pass 

    def square_laplacian(self, n):
        w = lat2W(n,n)
        a = np.zeros((n**2, n**2))
        np.fill_diagonal(a, 4.0)

        L = a - w.full()[0]
        #note c - stab(c) = Lv
        return(L)
    
    def generateidentity(self, dim):
        self.dim = dim 
        u = np.full(self.dim, ((len(self.dim) * 2)-1) * 2)
    
        z = stabilize(u)[0]
        print("stab(2u) calculated")
        p = np.full(self.dim, ((len(self.dim) * 2)-1) * 2)

        m = p - z 

        

        ans = stabilize(m)[0]
        return(ans)

    def create_visualization(self, a, name):
        a = a.astype('uint8')
        
        img = Image.fromarray(a, 'P')
        arr = np.array([255,255,255,0,0,204,102,0,102,0,0,0,255,0,0])  #0 = White, 1 = blue, 2 = purple, 3 = black
        
        img.putpalette(arr.tolist(), rawmode = 'RGB')

        img.save(name)


    def generate_data(self, size = 0, dim = (1,1), rand = False):
        
        data = [stabilize(np.random.randint(6, 10, dim), data = True)[1] for i in range(size)]

        return(data)

    def alternate(self, x, n, laplacian = None):
        
        if laplacian is None:
            laplacian = self.square_laplacian(5)

        x = x - n @ laplacian.reshape(n.shape)

        return(x)
if __name__ == "__main__":
    

    test = Sandpile()

    im_frame = Image.open('testimages\lena-image.png')
    np_frame = np.array(im_frame.getdata()) 



    print(np_frame.shape)
    print(np_frame)

    img = np_frame.reshape(256,256)


    plt.imshow(img, cmap = 'gray')
    plt.show()

    # alls = []
    # start_time = time.time()
    # for dim in range(50, 100):
    #     identity = test.generateidentity((dim,dim))
    #     #will rarely be equal to the identity matrix
    #     for i in range(50):
    #         target = np.random.randint(0,4, (dim,dim))
    #         perturbed, _ = stabilize(target + identity) 

    #         if np.all(perturbed == target):
    #              alls.append(np.array([target, 1], dtype = object))
    #             # alls.append(1)
    #         else:
    #             alls.append(np.array([target, 0], dtype = object))
    #             # alls.append(0)
    #     #will contain some recurrent matrices
    #     for i in range(50):
    #         target = np.random.randint(1,4, (dim,dim))
            
    #         perturbed, _ = stabilize(target + identity) 

    #         if np.all(perturbed == target):
    #              alls.append(np.array([target, 1], dtype = object))
    #             # alls.append(1)
    #         else:
    #             alls.append(np.array([target, 0], dtype = object))
    #             # alls.append(0)
    
    # alls = np.array(alls, dtype = object)
    
            

    # np.savez_compressed('recurrenceclassification.npz', alls)




































    

    

    






