import os
import cv2 as cv
with open("time.txt", "w") as time_file:

    e1 = cv.getTickCount()
    os.system("python TM_CCOEFF_NORMED/main.py")
    e2 = cv.getTickCount()
    time = (e2 - e1) / cv.getTickFrequency()
    time_file.write(f"TM_CCOEFF_NORMED: {time}s\n")

    e1 = cv.getTickCount()
    os.system("python YOLOv4-TM_CCOEFF_NORMED/main.py")
    e2 = cv.getTickCount()
    time = (e2 - e1) / cv.getTickFrequency()
    time_file.write(f"YOLOv4-TM_CCOEFF_NORMED: {time}s\n")

    e1 = cv.getTickCount()
    os.system("python YOLOv4/main.py")
    e2 = cv.getTickCount()
    time = (e2 - e1) / cv.getTickFrequency()
    time_file.write(f"YOLOv4: {time}s\n")
