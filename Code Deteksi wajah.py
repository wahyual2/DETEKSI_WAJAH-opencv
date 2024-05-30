#!/usr/bin/env python
# coding: utf-8

# 

# In[24]:


import cv2
from matplotlib import pyplot as plt

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('wahyu.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 10)  # Ketebalan garis: 10

# Convert BGR image to RGB for Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the output using Matplotlib
plt.imshow(img_rgb)
plt.title('Deteksi Wajah', loc='left')  # Judul di sebelah kiri
plt.axis('off')  # Tidak menampilkan sumbu
plt.show()


# In[ ]:




