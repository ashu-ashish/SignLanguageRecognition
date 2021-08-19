import cv2, pickle
import numpy as np
import os
import math
from keras.models import load_model

model = load_model('model/cnn_model_keras.h5')



def keras_process_image(img):
    img = cv2.resize(img, (50,50))
    img = np.array(img, dtype=np.uint8)
    img = np.reshape(img, (1,50,50, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text(pred_class):
    dictionary={0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',
                13:'N', 14:'Nothing', 15:'O',16:'P', 17:'Q', 18:'R', 19:'S', 20:'T',21:'U', 22:'V', 23:'W',
                24:'X', 25:'Y', 26:'Z'}
    return dictionary[pred_class]


def recognize():
    num_frames=0
    cap = cv2.VideoCapture(0)

    num_frames=0
    while(1):        
        ret,image = cap.read()
        image = cv2.flip(image, 1)
        if not ret:
            break
        k = cv2.waitKey(5) & 0xFF
        if k== ord('c'):
            num_frames=0
        if num_frames<=30:        
            background_image = image
            num_frames+=1
        if num_frames>30:
    #         cv2.imshow('background',background_image)
            current_image= image
            diff = cv2.absdiff(background_image,current_image)
            mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_blur = cv2.GaussianBlur(mask_thresh, (3, 3), 10)
            mask_erosion = cv2.erode(mask_blur, np.ones((5,5), dtype=np.uint8), iterations=1)
            mask_erosion = mask_erosion[150:400,150:400]
            cv2.imshow('final',mask_erosion)
            pred_probab, pred_class = keras_predict(model, mask_erosion)
#             print(pred_class)
            
                
            if pred_probab*100 > 80:
                    text = get_pred_text(pred_class)
                    print(text)
                    image = cv2.putText(image, text, (150, 100), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.rectangle(image, (150, 150), (400, 400), (255, 255,0), 3)   
            cv2.imshow('frame',image)



        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
recognize()
