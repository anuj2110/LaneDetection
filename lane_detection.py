import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import math
image=mpimage.imread("test_images/solidWhiteCurve.jpg")
plt.imshow(image)
plt.show()
print(image.shape)

def region_of_interest(img,vertices):
    mask=np.zeros_like(img)
    match_mask_color=255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image
height,width=image.shape[0],image.shape[1]
wd2,hd2=width/2,height/2
region_of_interest_vertices = [(0, height),(wd2, hd2),(width, height)]

roi=np.array([region_of_interest_vertices],dtype=np.int32)
gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
canny=cv2.Canny(gray_image,100,200)
plt.imshow(canny)
plt.show()
cropped= region_of_interest(canny,roi)
plt.imshow(cropped)
plt.show()
lines=cv2.HoughLinesP(cropped,rho=6,theta=np.pi/60,threshold=160,lines=np.array([]),minLineLength=40,maxLineGap=25)

def draw_lines(img,lines,color=[255,0,0],thickness=3):
    if lines is None :
        return
    copy=np.copy(img)
    line_image=np.zeros(shape=(copy.shape[0],copy.shape[1],3),dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),color,thickness)
    copy=cv2.addWeighted(img,0.8,line_image,1.0,0.0)
    return copy
line_image=draw_lines(image,lines)
plt.imshow(line_image)
plt.show()
left_line_x=[]
left_line_y=[]
right_line_x=[]
right_line_y=[]    

for line in lines:
    for x1,y1,x2,y2 in line:
        slope=(y2-y1)/(x2-x1)
        if math.fabs(slope)<0.5:
            continue
        if slope<=0:
            left_line_x.extend([x1,x2])
            left_line_y.extend([y1,y2])
        else:
            right_line_x.extend([x1,x2])
            right_line_y.extend([y1,y2])
min_y = image.shape[0] * (3 / 5) # <-- Just below the horizon
max_y = image.shape[0] # <-- The bottom of the image
poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))
left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))
poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))
right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))
line_image = draw_lines(
    image,
    [[
        [left_x_start, max_y, left_x_end, int(min_y)],
        [right_x_start, max_y, right_x_end, int(min_y)],
    ]],
    thickness=5,
)
plt.imshow(line_image)
plt.show()