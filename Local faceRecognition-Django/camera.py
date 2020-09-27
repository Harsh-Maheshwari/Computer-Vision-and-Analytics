import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import paths
import face_recognition
import pickle
import time
import os
import datetime

from faceApp.models import detectedData

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        return image        
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()

    def cap_frame(self,filename):
        success, image = self.video.read()
        cv2.imwrite(filename, image)
        return


def gen(camera):
    old_names = ['dcnwbjn']
    old_timestamp  = datetime.datetime(2011, 11, 4, 0, 0)
    data = pickle.loads(open("encodings.pickle", "rb").read())
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while True:
        frame = camera.get_frame()

        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 1)

        ########################### Logic ##########################

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)


        if len(names) > 0:

            directory=os.path.join('./Unknown') 
            isdir = os.path.isdir(directory)
            if not isdir:
                os.mkdir(directory)
            Unknown_count = len(next(os.walk(directory))[2]) 
            if names[0] == 'Unknown' and timestamp - old_timestamp > datetime.timedelta(seconds=5):
                cv2.imwrite('Unknown/'+str(Unknown_count)+'.png', frame)
                d = detectedData(employee_id=names[0],det_time=timestamp)
                d.save()
                old_names = names
                old_timestamp = timestamp
            if names[0] != 'Unknown':
                if names[0] != old_names[0]:
                    print(names)
                    d = detectedData(employee_id=names[0],det_time=timestamp)
                    d.save()
                    old_names = names
                    old_timestamp = timestamp
                elif timestamp - old_timestamp > datetime.timedelta(seconds=60):
                    d = detectedData(employee_id=names[0],det_time=timestamp)
                    d.save()
                    old_names = names
                    old_timestamp = timestamp

        ########################### Logic ##########################

        ret,frame  = cv2.imencode('.jpg', frame)
        frame = frame.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_emb():

    imagePaths = list(paths.list_images("dataset"))
    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,model="cnn")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    return

def cap_img(camera,directory):

    count = len(next(os.walk(directory))[2]) 
    camera.cap_frame(os.path.join(directory,str(count)+'.jpg'))

    return count
