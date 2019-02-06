import tkinter as tk  # for python 3
import numpy as np
import pygubu, cv2
from tkinter import filedialog, messagebox
from skimage import img_as_float, img_as_ubyte
from matplotlib import pyplot as plt

class Application:
    def __init__(self, master):

        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file('UI.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('OeH', master)

        builder.connect_callbacks(self)

    def GetVideo(self):
        path =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("avi files","*.avi"),("all files","*.*")))
        if(len(path) != 0):
            try:
                video = cv2.VideoCapture('{}'.format(path))
                self.video = video
            except:
                messagebox.showerror("Error", "Algo a salido mal, verifique que el contenido seleccionado sea una video")
    
    def GetImageNormal(self):
        path =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        if(len(path) != 0):
            try:
                img_normal = cv2.imread('{}'.format(path))
                self.img_normal = img_normal
            except:
                messagebox.showerror("Error", "Algo a salido mal, verifique que el contenido seleccionado sea una imagen") 

    def GetImageAltered(self):
        path =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        if(len(path) != 0):
            try:
                img_altered = cv2.imread('{}'.format(path))
                self.img_altered = img_altered
            except:
                messagebox.showerror("Error", "Algo a salido mal, verifique que el contenido seleccionado sea una imagen")

    def GetImage_Histogram(self):
        path =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))         
        if(len(path) != 0):
            try:
                img_histogram_grayscale = cv2.imread('{}'.format(path), 0)
                self.img_histogram_grayscale = img_histogram_grayscale
            except:
                messagebox.showerror("Error", "Algo a salido mal, verifique que el contenido seleccionado sea una imagen")

    def GetEqualizationGlobal(self):
        img_grayScale = np.copy(self.img_histogram_grayscale)
        equ = cv2.equalizeHist(img_grayScale)

        #Get Image Equalization
        cv2.namedWindow('Global', cv2.WINDOW_NORMAL)
        cv2.imshow('Global', equ)     
        cv2.waitKey(0) 

        #Get histogram
        plt.hist(equ.ravel(), 256, [0,256])
        plt.ylabel('Numero de pixeles')
        plt.xlabel("Nivel en grises")
        plt.title('Histrograma')
        plt.show()

    def GetEqualizationLocal(self):
        #input_limit
        #input_numberQuarter
        limit = float(self.builder.get_variable('input_limit').get())
        numberQuarter = int(self.builder.get_variable('input_numberQuarter').get())

        img_grayScale = np.copy(self.img_histogram_grayscale)
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(numberQuarter, numberQuarter))
        cl1 = clahe.apply(img_grayScale) 
        cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Imagen Original', self.img_histogram_grayscale) 
        cv2.namedWindow('Ecualización Local Adaptativa', cv2.WINDOW_NORMAL)
        cv2.imshow('Ecualización Local Adaptativa', cl1)  

        #Get histogram
        plt.hist(cl1.ravel(), 256, [0,256])
        plt.ylabel('Numero de pixeles')
        plt.xlabel("Nivel en grises")
        plt.title('Histrograma')
        plt.show()  

    def StartImageDifference(self):
        imgNormal = np.array(self.img_normal , dtype=np.uint8)  
        imgAltered = np.array(self.img_altered , dtype=np.uint8)      
        img = np.array(self.img_normal, dtype=np.uint8)

        img_aligned_gray = cv2.cvtColor(self.AlignmentImage(imgNormal, imgAltered), cv2.COLOR_BGR2GRAY)
        img_normal_gray = cv2.cvtColor(imgNormal, cv2.COLOR_BGR2GRAY)

        row, col = img_normal_gray.shape

        for y in range(0, row):
            for x in range(0, col):
                difference = img_normal_gray[y, x] - img_aligned_gray[y, x]

                if( difference != 0):
                    if(((img_normal_gray[y, x] - img_aligned_gray[y, x]) <= 10 ) or ((img_aligned_gray[y, x] - img_normal_gray[y, x]) <= 10)):
                        img[y, x] = 0
                    else: 
                        img[y, x] = 255
                else:
                    img[y, x] = 0

        cv2.namedWindow('Image_normal', cv2.WINDOW_NORMAL)
        cv2.imshow('Image_normal', self.img_normal) 

        cv2.namedWindow('Image_altered', cv2.WINDOW_NORMAL)
        cv2.imshow('Image_altered', self.img_altered) 

        cv2.namedWindow('Image_search', cv2.WINDOW_NORMAL)
        cv2.imshow('Image_search', img)     
        cv2.waitKey(0) 

    def StartErasyNoisy(self):
        video = self.video
        if (video.isOpened()== False): 
            messagebox.showerror("Error", "The video can't be open")
        else:
            T = 1
            limit_T = int(self.builder.get_variable('IDFrame').get())

            # Read until video is completed
            while(video.isOpened()):
                # Capture frame-by-frame
                ret, frame = video.read()
                
                #Get x, y, channels of frame
                value_x, value_y, channels = frame.shape

                if ret == True:
                    if(T > 1):
                        #Alignment image
                        alignment =  self.AlignmentImage(np.array( promFrame_alignment , dtype=np.uint8), np.array( frame , dtype=np.uint8))

                        firstOperation = float((T - 1) / T)

                        for x in range(0, value_x):
                            for y in range(0, value_y):       
                                #Remove erase noisy
                                secondOperation_blue = ((1/ T) * alignment.item(x, y, 0))
                                secondOperation_red = ((1/ T) * alignment.item(x, y, 1))
                                secondOperation_green = ((1/ T) * alignment.item(x, y, 2))

                                resultOperation_blue = (firstOperation * promFrame_alignment.item(x, y, 0)) + secondOperation_blue
                                resultOperation_red = (firstOperation * promFrame_alignment.item(x, y, 1)) + secondOperation_red
                                resultOperation_green = (firstOperation * promFrame_alignment.item(x, y, 2)) + secondOperation_green
                                
                                promFrame_alignment.itemset((x, y, 0), resultOperation_blue)
                                promFrame_alignment.itemset((x, y, 1), resultOperation_red)
                                promFrame_alignment.itemset((x, y, 2), resultOperation_green)
                    

                        if(T == limit_T):
                            promFrame_alignment = np.array(promFrame_alignment , dtype=np.uint8)
                            cv2.namedWindow('Image Normal')
                            cv2.imshow('Image Normal', frame)     

                            cv2.namedWindow('Image Aligned')
                            cv2.imshow('Image Aligned', promFrame_alignment)     
                            cv2.waitKey(0) 
                            break                         
                    else:
                        promFrame_alignment = frame
                        
                    T = T + 1
                    # Break the loop
                else: 
                    break
            
            
            print("Sali")

            # When everything done, release the video capture object
            video.release()
            # Closes all the frames
            cv2.destroyAllWindows()

    def ShowImage_Histogram(self):     
        try:
            img_histogram_grayscale = np.copy(self.img_histogram_grayscale)
            plt.hist(img_histogram_grayscale.ravel(), 256,[0,256])
            plt.ylabel('Numero de pixeles')
            plt.xlabel("Nivel en grises")
            plt.title('Histograma')
            plt.show()
        except:
            messagebox.showerror("Error", "Algo a salido mal, verique sus datos")  
    
    def AlignmentImage(self, promFrame, frame):
        im1_gray = cv2.cvtColor(promFrame,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        sz = promFrame.shape

        warp_mode = cv2.MOTION_TRANSLATION

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        number_of_iterations = 10000; 

        termination_eps = 1e-10;

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (frame, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
        # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(frame, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
        return im2_aligned

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.mainloop()