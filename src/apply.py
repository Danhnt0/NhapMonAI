
import cv2
import numpy as np
import tensorflow as tf
import pickle 

image = cv2.imread("images/test2.jpg")
# image = cv2.resize(image,(300,200))
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
        cv2.putText(copy,str(int(knn_pred)),(x,y-80),0,1,(0,255,0),2)

        softmax_pred = softmax.predict(subImage.reshape(1,784))
        cv2.putText(copy,str(int(softmax_pred)),(x,y-40),0,1,(0,255,255),2)


cv2.putText(copy,"CNN",(50,30),0,0.5,(255,0,0),1)
cv2.putText(copy,"KNN",(50,45),0,0.6,(0,255,0),1)
cv2.putText(copy,"Softmax",(50,60),0,0.5,(0,255,255),1)
cv2.putText(copy,"DNN",(50,75),0,0.5,(0,0,255),1)

cv2.imshow("image",copy)
cv2.waitKey(0)


   
