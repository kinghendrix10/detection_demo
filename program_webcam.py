import sys
import cv2

# Creating a face cascade
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
# Set the video source to the default webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame
    ret, frame = video_capture.read()

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    faces = faceCascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (column, row, width, height) in faces:
        cv2.rectangle(frame, (column, row),(column+width, row+height),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everthing is done, release the capture
video_capture.release()
cv2.destroyAllWindows()