from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras_preprocessing.image import img_to_array
#from tensorflow.keras.utils import img_to_array
from keras.models import load_model

def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq

def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    print("Emotion: ",label)
    if label in ['scared','sad','angry','disgust']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return label
    
def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    print("Prediction",stress_value)
    if stress_value>=75:
        return stress_value,"High Stress"
    else:
        return stress_value,"low_stress"
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:\major project code\Stress-Detection-master(run)\shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("E:\major project code\Stress-Detection-master(run)\_mini_XCEPTION.102-0.66.hdf5", compile=False)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

points = []
while(True):
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=800,height=800)
    
    
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

    #preprocessing the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray,0)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for detection in detections:
        emotion = emotion_finder(detection,gray)
        cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
           
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        #cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        stress_value,stress_label = normalize_values(points,distq)
        cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("Stress value: ",stress_label)
        print("\n\n")
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
plt.plot(range(len(points)),points,'ro')
plt.title("Stress Levels")
plt.show()
