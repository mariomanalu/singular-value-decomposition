# Singular Value Decomposition: How to Store any Image
## Introduction
Digital images are made up of the smallest graphical elements capable of storing one color at a time called pixels. The collection of pixels that make up an image are stored as a matrix where each entry contains one pixel. Grayscale images, for example, can be represented by a matrix with each entry containing a number between 0 (black) and 255 (white) inclusively. Color images, in turn, can be represented by three matrices. Each matrix represents the intensity of red, green, and blue in each pixel. Let us assume that our desktop background is an image of size 1280 * 1024. If it was a grayscale image, our computers would need to store 1280 * 1024 = 1310720 pixel values. If it was a color image, our computers would need to store 1310720 red pixel values, 1310720 green pixel values, and 1310720 blue pixel values. That is a total of **3,932,160 ** different pixel values. If one pixel takes up one byte of memory, then our computers need about **3.93 MB** of memory to store that image alone. The size of my entire numerical software package unzipped is only 45 KB, so 3.93 MB is a significant amount of memory. If your computer likes to run out of memory, I know mine certainly does, it would be wise for us to compress that background image before storing it in the memory. 

Image compression minimizes the size of an image by decomposing the matrix and eliminating the entries that are relatively small. Since some of the entries are being zeroed-out, the compressed image will be a less-sharp, less-detailed version of the original image. We can tune the quality of the compressed image so that it does not exceed the amount of memory we are willing to allocate. There is a direct relationship between the size of the compressed image and its quality. Lower size will result in lower quality of the compressed image. In most cases, though, we want to compress an image in such a way that our eyes will hardly be able to distinguish the compressed image from the original image.

## Basic requirements
1. Matplotlib installed. To install on MacOS, run python3 -m pip install matplotlib.
2. PIL/Pillow installed. To install on MacOS, run python3 -m pip install --upgrade Pillow
## How to use
1. Clone the repo or download svd.py and svd-test.py separately.
2. In line 10 of svd-test.py, assign with the path to the image that you want to decompress.
3. In line 34 of svd-test.py, assign numberOfSingularValues with an integer of your choosing. Greater integer implies greater image quality, which also means, greater size of image after decompression. This is the trade-off. Therefore, pick an integer that does not reduce the image quality by a lot and still outputs an image of size smaller than the original image.

## Further reading
Read the the underlying mathematical computation behind image decompression in the pdf file.
