import cv2
import speech_recognition as sr
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import numpy as np

# Initialize face detection cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize video capture and microphone
cap = cv2.VideoCapture(0)
microphone = sr.Microphone()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w * h < 500:  # Skip small face detections
            continue

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            emotion_text_position = (x, y)
            cv2.putText(frame, label, emotion_text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        with microphone as source:
            print("Say something:")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)

            if "hello" in text.lower() and ("how r u" in text.lower() or "how are you" in text.lower()):
                response = "Initiating Talk"
                response_position = (x, y +30)  # Position captioning text above emotion text
                cv2.putText(frame, response, response_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                print("Bot: " + response)
            elif "what" in text.lower() or ("how" in text.lower()):
                response = "Asking question"
                response_position = (x, y +30)  # Position captioning text above emotion text
                cv2.putText(frame, response, response_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                print("Bot: " + response)
            elif "bye" in text.lower():
                response = "Conversation end"
                response_position = (x, y +30)  # Position captioning text above emotion text
                cv2.putText(frame, response, response_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                print("Bot: " + response)
            elif "greetings" in text.lower() and ("shall" in text.lower() or "should" in text.lower()) :
                response = "initiate dialogue"
                response_position = (x, y +30)  # Position captioning text above emotion text
                cv2.putText(frame, response, response_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                print("Bot: " + response)
            else:
                response = text

              # Position emotion text below captioning text

        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio.")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))

    cv2.imshow('Face Recognition with Live Captioning', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
