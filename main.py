from random import randrange
import cv2
from cv2 import waitKey
from cv2 import CascadeClassifier

if __name__ == '__main__':
    #use pretrained model from open-cv
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    while True:
        successfull_fream_read, frame = cam.read()
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FaceDetection
        face_coord = trained_face_data.detectMultiScale(gray_scale)
        
        #Draw Rectangle
        for (x,y,w,h) in face_coord:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

        res = cv2.imshow('FaceDetectorApp', frame)
        key = cv2.waitKey(1)

        if key==81 or key == 113:
            break

    cam.release()
    print("Finished")
    
