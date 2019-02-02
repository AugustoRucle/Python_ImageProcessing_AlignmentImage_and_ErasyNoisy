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

        # create a CLAHE object (Arguments are optional).
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

    def StartImageRegistration(self):
        im2 = np.array(self.img_normal , dtype=np.uint8)  
        im1 = np.array(self.img_altered , dtype=np.uint8)      

        im1Reg = self.myFunction(im2, im1)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        row, col = im1Reg.shape

        for y in range(0, row):
            for x in range(0, col):
                d1 = im2[y, x] - im1Reg[y, x]

                if( d1 != 0):
                    if(((im2[y, x] - im1Reg[y, x]) <= 10 ) or ((im1Reg[y, x] - im2[y, x]) <= 10)):
                        im2[y, x] = 0
                    else: 
                        im2[y, x] = 255
                else:
                    im2[y, x] = 0

        cv2.namedWindow('Image_normal', cv2.WINDOW_NORMAL)
        cv2.imshow('Image_normal', self.img_normal) 

        cv2.namedWindow('Image_altered', cv2.WINDOW_NORMAL)
        cv2.imshow('Image_altered', self.img_altered) 

        cv2.namedWindow('Image_search', cv2.WINDOW_NORMAL)
        cv2.imshow('Image_search', im2)     
        cv2.waitKey(0) 


    #Start 
    def StartErasyNoisy(self):
        video = self.video
        if (video.isOpened()== False): 
            messagebox.showerror("Error", "The video can't be open")
        else:
            T = 1
            promFrame = []
            limit_T = int(self.builder.get_variable('IDFrame').get())

            # Read until video is completed
            while(video.isOpened()):
                # Capture frame-by-frame
                ret, frame = video.read()
                if ret == True:
                    if(T > 1 and T < limit_T):
                        #Alignment image
                        promFrame = self.AlignmentImage(np.array(promFrame , dtype=np.float32), np.array(frame , dtype=np.float32))
                        #Remove erase noisy
                        firstOperation = float((T - 1) / T)
                        secondOperation = ((1 / T) * frame)
                        promFrame = (firstOperation * promFrame)+ secondOperation
                    elif(T == limit_T):
                        #Alignment image
                        promFrame = self.AlignmentImage(np.array(promFrame , dtype=np.float32), np.array(frame , dtype=np.float32))
                        #Get PromFrame
                        firstOperation = float((T - 1) / T)
                        secondOperation = ((1 / T) * frame)
                        promFrame = (firstOperation * promFrame)+ secondOperation  
                        promFrame = np.array(promFrame , dtype=np.uint8)
                        #Show image
                        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                        cv2.imshow('Image', promFrame)     
                        cv2.waitKey(0)                               
                    elif(T == 1):
                        promFrame = frame

                    T = T + 1
                    # Break the loop
                else: 
                    break

            # When everything done, release the video capture object
            video.release()
            # Closes all the frames
            cv2.destroyAllWindows()

    def ShowImage_Histogram(self):     
        try:
            img_histogram_grayscale = np.copy(self.img_histogram_grayscale)
            #print("ArrayUno: ", img_histogram_grayscale)
            #hist,bins = np.histogram(img_histogram_grayscale.ravel(), 256, [0,256])
            #print("ArrayDos: ", img_histogram_grayscale.ravel())
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