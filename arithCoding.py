import numpy as np
import cv2
from queue import PriorityQueue

img = cv2.imread('bika.png', 0)
rows, cols = img.shape

blockSize = int(input("Enter a Block Size value: "))
floatType = input("Enter Float type(16, 32, 64): ")
lengthDict = {}
frequency = 1

arr = np.array([0])

for i in range(rows):
    for j in range(cols):
        arr = np.append(arr, img[i, j] - img[i, j] % 3)

arr = np.delete(arr, 0)

while arr.size % blockSize != 0:
    arr = np.append(arr, 0)

unique, counts = np.unique(arr, return_counts=True)
lengthDict = dict(zip(unique, counts))
startDict = dict()
accLength = 0

for i in lengthDict:
    lengthDict[i] /= img.size
    startDict[i] = accLength
    accLength += lengthDict[i]

probabilitiesArr = np.zeros(256)

for i in lengthDict:
    probabilitiesArr[i] = lengthDict[i]

np.save("Probabilities Array", probabilitiesArr)

if floatType == "16":
    encodedArr = np.zeros(int(arr.size / blockSize), np.float16)
elif floatType == "32":
    encodedArr = np.zeros(int(arr.size / blockSize), np.float32)
else:
    encodedArr = np.zeros(int(arr.size / blockSize), np.float64)

start = 0
length = frequency
k = 0
f = 0
for i in arr:
    if f % blockSize == 0:
        if f != 0:
            encodedArr[k] = start + length / 2
            k += 1
        start = 0
        length = frequency
    start += startDict[i] * length
    length = length * lengthDict[i]
    f += 1
encodedArr[k] = start + length / 2

np.save("Encoded Image", encodedArr)
decodedArr = np.zeros(arr.size, int)


############################################ DECODING ################################################

lengthDict = dict()
startDict = dict()

for i in range(0, probabilitiesArr.size):
    if probabilitiesArr[i] != 0:
        lengthDict[i] = probabilitiesArr[i]

accLength = 0
for i in lengthDict:
    startDict[i] = accLength
    accLength += lengthDict[i]

start = 0
length = frequency
k = 0
for i in encodedArr:
    for f in range(0, blockSize):
        for j in startDict:
            if i < ((lengthDict[j] + startDict[j]) * length + start):
                decodedArr[k] = j
                k += 1
                start += startDict[j] * length
                length = length * lengthDict[j]
                break
    start = 0
    length = frequency

k = 0
for i in range(rows):
    for j in range(cols):
        img[i, j] = decodedArr[k]
        k += 1

cv2.imshow('imag', img)
cv2.imwrite("Decoded Bika.png", img)
cv2.waitKey(0)
