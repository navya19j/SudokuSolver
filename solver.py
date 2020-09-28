from cv2 import cv2
import numpy as np
import matplotlib as plt
from imutils.perspective import four_point_transform, order_points
from tensorflow.keras.models import load_model
from skimage.segmentation import clear_border
from algo import *

#IMAGE PROCESSING
img = cv2.imread("test_2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(9,9),0)
#Adaptive gaussian thresholding produced the best results
blu_thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,11,8)

inv = cv2.bitwise_not(blu_thresh)
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
fin = cv2.dilate(inv,kernel)


##DETECTING THE GRID
height,width,_ = img.shape
maxi = -1
for x in range (0,height):
    for y in range (0,width):
        if fin[x,y] >=128 :
            area=cv2.floodFill(fin,None,(y,x),80)[0]
            if area >maxi :
                maxpt = (y,x)
                maxi = area

cv2.floodFill(fin,None,maxpt,255)
# cv2.imshow("a", fin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
for x in range (height):
    for y in range (width):
        if fin[x,y] == 80 and x!=maxpt[1] and y!=maxpt[0] :
            cv2.floodFill(fin,None,(y,x),0)
# fin = cv2.resize(fin, (28, 28))
# cv2.imshow("a", fin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
fin_erode = cv2.erode(fin,kernel)
# cv2.imwrite("a1.jpg", fin_erode)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# count,_ = cv2.findContours(fin_erode.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# count = sorted(count, key=cv2.contourArea, reverse=True)
# # cv2.drawContours(img, count, -1, (0, 255, 0), 3) 

cnts, _ = cv2.findContours(fin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# sort contours decreasing order area wise
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
mask = np.zeros((fin.shape), np.uint8)
c = cnts[0]

clone = img.copy()

peri = cv2.arcLength(c, closed=True)
poly = cv2.approxPolyDP(c, epsilon=0.02 * peri, closed=True)

if len(poly) == 4:
    cv2.drawContours(clone, [poly], -1, (0, 0, 255), 2)
    warped = four_point_transform(img, poly.reshape(-1, 2))

model = load_model("test_model.h5")
# gray_warp = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
# blurred_warp = cv2.GaussianBlur(gray_warp,(5,5),0)
# #Adaptive gaussian thresholding produced the best results
# thresh_warp = cv2.adaptiveThreshold(blurred_warp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,1)
# inv_warp_1 = cv2.bitwise_not(thresh_warp)
# inv_warp = cv2.dilate(inv_warp_1,kernel)
# cv2.imshow("a", inv_warp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# conts, _ = cv2.findContours(inv_warp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(thresh_warp,conts,-1,(0,255,0),3)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
height_fin,width_fin = warped.shape
height_cut = int(height_fin/9)
width_cut = int(width_fin/9)
labels=[]
centers=[]
for i in range (0,9):
    for j in range (0,9):
        crop = warped[i*height_cut:(i+1)*height_cut,j*width_cut:(j+1)*width_cut]
        digit = cv2.resize(crop, (28, 28))
        _, digit2 = cv2.threshold(digit, 80, 255, cv2.THRESH_BINARY_INV)
        digit3 = clear_border(digit2)
        numpixel = cv2.countNonZero(digit3)
        _, digit4 = cv2.threshold(digit3, 0, 255, cv2.THRESH_BINARY_INV)
        if numpixel < 3 :
            label = 0
        else:
            _, digit4 = cv2.threshold(digit4, 0, 255, cv2.THRESH_BINARY_INV)
            digit4 = digit4 / 255.0
            array = model.predict(digit4.reshape(1, 28, 28, 1))
            label = np.argmax(array)
        labels.append(label)
        centers.append((i*height_cut/2,j*width_cut/2))
        # print(label)
        # cv2.imshow("a", digit4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# cv2.imshow("a", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(labels)
# # grid = []
# # for i in range (0,len(labels),9):
# #     grid.append(labels[i:i+9])

# print(grid)
# # if(solve_sudoku(grid)): 
# #     printMatrix(grid) 
# # else: 
# #     print ("No solution exists")

# font = cv2.FONT_HERSHEY_SIMPLEX 
# fontScale = 1
# color = (255, 0, 0) 
# thickness = 2

# for i in  range (0,centers):

# image = cv2.putText(image, , org, font,  
#                    fontScale, color, thickness, cv2.LINE_AA) 
grid = np.array(labels).reshape(9, 9)
print(grid)

# soduko = solve_sudoku(grid)
# zero_indices = zip(*np.where(grid == 0))
# zero_centres = np.array(centers).reshape(9, 9, 2)

# if soduko==True:
#     warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     for row, col in zero_indices:
#         cv2.putText(
#             warped,
#             str(grid[row][col]),
#             (zero_centres[row][col][0] - 10, zero_centres[row][col][1] + 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             2,
#             (0, 0, 255),
#             3,
#         )
#     cv2.imshow("Solved", warped)
#     cv2.waitKey(0)




