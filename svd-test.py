import os
import numpy as np
from PIL import Image
from svd import svd
import matplotlib.pyplot as plt
#%matplotlib inline
# Test file for image reduction

#Load image of Waldo the dog
path = 'WaldoasWaldo.jpg'
img = Image.open(path)
s = float(os.path.getsize(path))/1000
#Print the size of the image
print("Size(dimension): ",img.size)
#plt.title("Original Image (%0.2f Kb):" %s)
#Show the image
#plt.imshow(img)

#Convert the image into matrix
imggray = img.convert('LA')
imgmat = np.array( list(imggray.getdata(band = 0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure()
#plt.imshow(imgmat, cmap = 'gray')
#plt.title("Image after converting it into the Grayscale pattern")
#plt.show()

print("After compression: ")
#Execute the Singular Value Decomposition process 
U, S, Vt = np.linalg.svd(imgmat) 

#NOTE: One can change the numberOfSingularValues as one wish. Greater number means greater quality.
numberOfSingularValues = 5
cmpimg = np.matrix(U[:, :numberOfSingularValues]) * np.diag(S[:numberOfSingularValues]) * np.matrix(Vt[:numberOfSingularValues,:])
plt.imshow(cmpimg, cmap = 'gray')
plt.show()
result = Image.fromarray((cmpimg ).astype(np.uint8))
imgmat = np.array( list(img.getdata(band = 0)), float)
imgmat.shape = (img.size[1], img.size[0])
imgmat = np.matrix(imgmat)
plt.figure()
plt.imshow(imgmat)
plt.show()
