#Importing all the important models and install them if not installed on your device
#computer vision used for all purpose camera activity app
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# to convert png/jpeg into the format which interpretor can understand
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Setting an HTTPS Context to fetch data from OpenML
#secure socket layer (ssl)
#if environment is not created then also run the program on different operating system
#itis like sandbox as it doesn't affect os and is temprory setup
#its like a virtual stage which doesn't affect on os
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

#Fetching the data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#scaling the features turning to gray scale
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

0#Calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)

#Starting the camera
#open cv library help to capture video through webcam and can perform desired operation on video and control webcam
#function videocapture 0 depicts number of camera which means first camera and if number of camera installed then use 1,2,3,4 as number of camera you want to use
cap = cv2.VideoCapture(0)

#infinite loop
while(True):
  # Capture frame-by-frame
  try:
    #ret is flag if frame is read it will be true else false
    ret, frame = cap.read()

    # Our operations on the frame come here
    #print formate is CMYK (cyan,magenta,yellow,black)
    #normal A4 size printing RGB 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video so that software recognize only to that small area instead of whole screen
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    #cv2.rectangle(sourcewhose image is to be created,width,height, colour of rim,thickness)
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest //focus area
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #Converting cv2 image to pil format
    im_pil = Image.fromarray(roi)

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    #1 millisec 0xFF is hexadecimal code for 255 highest ASCII value
    #ASCII value should not fall beyond 255 and here q is ASCII value to quit the video
    # q for quit and this is written to stop infinite loop
    #q when pressed will help to close the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
