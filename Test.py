import tensorflow as tf
import cv2
import numpy as np


model = tf.keras.models.load_model("TrainedModel")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

previous = 0

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        image=frame, scaleFactor=1.3, minNeighbors=4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            crop = frame[y: y+h, x: x + w]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (64, 64))

            crop = np.array(crop).reshape((-1, 64, 64, 3))

            pred = np.argmax(model.predict(crop)[0])

            if pred == 0:
                cv2.putText(frame, "You are wearing a mask", (360, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                previous = 1
            else:
                cv2.putText(frame, "please wear a mask", (360, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

                previous = 0

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No face dected", (360, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

        previous = -1

    cv2.imshow("Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
