import tkinter as tk
from tkinter import filedialog
import cv2
import tensorflow as tf
import pickle
import numpy as np
import PIL.Image, PIL.ImageTk


window = tk.Tk()
window.title('Detect number')
window.geometry('900x600')


def open_file():
    # get the file path from the user with format .png, .jpg, .jpeg
    file_path = filedialog.askopenfilename(filetypes=[('Image Files', ['.png', '.jpg', '.jpeg'])])

    detect(file_path)

btn = tk.Button(window, text='Choosen File', command=open_file)
# set center position
btn.place(relx=0.5, rely=0.5, anchor='center')
btn.pack()

label = tk.Label(window, text='Please open a file',font=('Arial', 20))

# hide the label
label.pack_forget()

def detect(file_path):
    image = cv2.imread(file_path)
    copy = image.copy()

    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
    _,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    #load model
    model = tf.keras.models.load_model("model/model.h5")
    C_model = tf.keras.models.load_model("model/C_model.h5")
    knn = pickle.load(open('model/knn.pkl', 'rb'))
    softmax = pickle.load(open('model/softmax.pkl', 'rb'))




    for i in contours:
    # if cv2.contourArea(i) > 400:
            (x,y,w,h) = cv2.boundingRect(i)
            cv2.rectangle(copy,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),1)
            subImage = thre[y-10:y+h+10,x:x+w]
            pad = abs(subImage.shape[0] - subImage.shape[1])
            # add pad to the image
            if subImage.shape[0] > subImage.shape[1]:
                subImage = np.pad(subImage,((0,0),(pad//2,pad//2)),'constant',constant_values=(0,0))
            else:
                subImage = np.pad(subImage,((pad//2,pad//2),(0,0)),'constant',constant_values=(0,0))

            # subImage = thre[y:y+h,x:x+w]
            # subImage = np.pad(subImage,(20,20),'constant',constant_values=(0,0))
            

            subImage = cv2.resize(subImage, (28, 28), interpolation=cv2.INTER_AREA)

            img = subImage.reshape(1,28,28,1)
            img = img/255.0
            img = img.astype(np.float32)
            
            pred = model.predict(img)
            pred = np.argmax(pred)
            cv2.putText(copy,str(int(pred)),(x,y+120),0,1,(0,0,255),2)

            CNN_pred = C_model.predict(img)
            CNN_pred = np.argmax(CNN_pred)
            cv2.putText(copy,str(int(CNN_pred)),(x,y+160),0,1,(255,0,0),2)

            knn_pred = knn.predict(subImage.reshape(1,784))
            cv2.putText(copy,str(int(knn_pred)),(x,y+200),0,1,(0,255,0),2)

            softmax_pred = softmax.predict(subImage.reshape(1,784))
            cv2.putText(copy,str(int(softmax_pred)),(x,y+240),0,1,(0,255,255),2)

    #set image to label
            show_img = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
            show_img = cv2.resize(show_img, (500, 300), interpolation=cv2.INTER_AREA)
            # set background of label by show_img
            show_img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(show_img))
            label.configure(image=show_img)
            label.image = show_img
            label.pack()
            window.update()




window.mainloop()