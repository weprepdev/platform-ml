import cv2
import numpy as np
from time import sleep
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing.image import img_to_array
import imutils

def analyze_video(video_path):
    def preprocess_input(image):
        image = cv2.resize(image, (img_width, img_height))  # Resizing images for the trained model
        ret = np.empty((img_height, img_width, 3))
        ret[:, :, 0] = image
        ret[:, :, 1] = image
        ret[:, :, 2] = image
        x = np.expand_dims(ret, axis = 0)
        x -= 128.8006   # np.mean(train_dataset)
        x /= 64.6497    # np.std(train_dataset)
        return x

    # Model used
    train_model =  "ResNet"
    img_width, img_height = 197, 197
    emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


    model = load_model('models/ResNet-50.h5')
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')
    distract_model = load_model('models/distraction_model.hdf5', compile=False)


    frame_w = 1200
    border_w = 2
    min_size_w = 240
    min_size_h = 240
    min_size_w_eye = 60
    min_size_h_eye = 60
    scale_factor = 1.1
    min_neighbours = 5
    Engage = list()
    EmotionOrder = list()

    video_capture = cv2.VideoCapture(video_path)

    cntTime = 0
    cntFear = 0
    cntAnger = 0
    cntDisgust = 0
    cntNeutral = 0
    cntSadness = 0
    cntSurprise = 0
    cntHappiness = 0
    emotion_orders = {}

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    subsampling_rate = 5

    frame_count = 0

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
        else:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            print(frame_count)


            if frame_count % subsampling_rate != 0:

                continue

            frame = imutils.resize(frame, width=frame_w)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
                # image:		Matrix of the type CV_8U containing an image where objects are detected
                # scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
                # minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
                # minSize:		Minimum possible object size. Objects smaller than that are ignored

            faces = faceCascade.detectMultiScale(
                gray_frame,
                scaleFactor		= scale_factor,
                minNeighbors	= min_neighbours,
                minSize			= (min_size_h, min_size_w))

            if len(faces) == 0:
                if cntTime == 0:
                    emotion_orders[cntTime] = {
                        "Anger": 0.0833333,
                        "Disgust": 0.0833333,
                        "Fear": 0.0833333,
                        "Happiness": 0.0833333,
                        "Sadness": 0.0833333,
                        "Surprise": 0.0833333,
                        "Neutral":0.5
                    }
                    Engage.append(np.random.uniform(0.0, 0.2))
                else:
                    emotion_orders[cntTime] = emotion_orders.get(cntTime - 1)
                    Engage.append(Engage[cntTime - 1] * 0.3)
                cntTime += 1
                continue

            prediction = None
            x, y = None, None

            for (x, y, w, h) in faces:

                ROI_gray = gray_frame[y:y+h, x:x+w]
                ROI_color = frame[y:y+h, x:x+w]
                # Draws a simple, thick, or filled up-right rectangle
                    # img:          Image
                    # pt1:          Vertex of the rectangle
                    # pt2:          Vertex of the rectangle opposite to pt1
                    # rec:          Alternative specification of the drawn rectangle
                    # color:        Rectangle color or brightness (BGR)
                    # thickness:    Thickness of lines that make up the rectangle. Negative values, like CV_FILLED ,
                    #               mean that the function has to draw a filled rectangle
                    # lineType:     Type of the line
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                emotion = preprocess_input(ROI_gray)
                prediction = model.predict(emotion)
                print(emotions[np.argmax(prediction)] + " predicted with accuracy " + str(max(prediction[0])))
                top = emotions[np.argmax(prediction)]

                eyes = eye_cascade.detectMultiScale(ROI_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))


                probs = list()

                # loop through detected eyes
                for (ex,ey,ew,eh) in eyes:
                    # draw eye rectangles
                    cv2.rectangle(ROI_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # get colour eye for distraction detection
                    roi = ROI_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
                    # match CNN input shape
                    roi = cv2.resize(roi, (64, 64))
                    # normalize (as done in model training)
                    roi = roi.astype("float") / 255.0
                    # change to array
                    roi = img_to_array(roi)
                    # correct shape
                    roi = np.expand_dims(roi, axis=0)

                    # distraction classification/detection
                    pred = distract_model.predict(roi)
                    # save eye result
                    probs.append(pred[0])

                # get average score for all eyes
                probs_mean = np.mean(probs)

                # get label
                if probs_mean <= 0.5:
                    label = 'distracted'
                else:
                    label = 'focused'

                #increase cnt
                if top == 'Anger':
                    cntAnger += 1
                if top == 'Disgust':
                    cntDisgust += 1
                if top == 'Fear':
                    cntFear += 1
                if top == 'Happiness':
                    cntHappiness += 1
                if top == 'Sadness':
                    cntSadness += 1
                if top == 'Surprise':
                    cntSurprise += 1
                if top == 'Neutral':
                    cntNeutral += 1

                prediction_list = prediction[0].tolist()
                emotion_orders[cntTime] = {
                    "Anger": prediction_list[0],
                    "Disgust": prediction_list[1],
                    "Fear": prediction_list[2],
                    "Happiness": prediction_list[3],
                    "Sadness": prediction_list[4],
                    "Surprise": prediction_list[5],
                    "Neutral": prediction_list[6]
                }

                #increse time
                cntTime += 1

                #Append engagement level
                if(np.isnan(probs_mean)):
                    Engage.append(np.random.uniform(0.6,0.7))
                else:
                    Engage.append(probs_mean)

                text = top + ' + ' + label
                cv2.putText(frame, text, (x, y+(h+50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




    #Engagement Level Graph
    time = np.arange(cntTime + 1)
    engm = np.asarray(Engage)
    engagement = {}

    for tim, eng in zip(time, engm):
        engagement[int(tim)] = str(eng)

    labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    values = [cntAnger, cntDisgust, cntFear, cntHappiness, cntSadness, cntSurprise, cntNeutral]
    emotions = {}
    for lab, val in zip(labels, values):
        emotions[lab] = str(val)

    new_emo_ord = {"emotion_orders": emotion_orders}

    new_eng = {"engagement": engagement}
    new_emo = {"emotions": emotions}

    result = {}
    result.update(new_eng)
    result.update(new_emo_ord)
    result.update(new_emo)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    return result
