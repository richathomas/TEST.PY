import cv2
import os
cam = cv2.VideoCapture(0)
path="E:/dataset"
os.mkdir(path)
os.chdir(path)
cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    img_name = "opencv_frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(9)

    if k%256 == 9:
        # ESC pressed
        print("Tab pressed, closing...")
        break
cam.release()
