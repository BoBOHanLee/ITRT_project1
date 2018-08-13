import cv2
import numpy as np
import functions as fn
img1 =cv2.imread('normal.jpg',0)
#test

#roi
img1_process=img1[0:np.shape(img1)[0],11:109]
img1_show = cv2.cvtColor(img1_process, cv2.COLOR_GRAY2BGR)


thresh=fn.Inhence_and_threshod(img1_process,111)
image1, contours1, hierarchy = cv2.findContours(                          #找最外層的輪廓
    fn.inner_fill(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mor1_color = cv2.cvtColor(fn.inner_fill(thresh), cv2.COLOR_GRAY2BGR)
img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
fn.draw(contours1,img1_color)
fn.find_roi_coordinate(contours1,img1_color)



tmp1 = np.hstack((img1_show,mor1_color,img1_color))

cv2.imshow("normal",tmp1)

k = cv2.waitKey(0)
if k==ord('ｑ'):
    cv2.destroyAllWindows()


