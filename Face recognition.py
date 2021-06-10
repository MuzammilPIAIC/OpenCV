import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

faces  = face_cascade.detectMultiScale(gray, 1.3, 5)

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
 
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
   # Captures video_capture frame by frame
    _, frame = video_capture.read()
 
    # To capture image in monochrome                   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
     
    # calls the detect() function   
    canvas = detect(gray, frame)  
 
    # Displays the result on camera feed                    
    cv2.imshow('Video', canvas)
 
    # The control breaks once q key is pressed                       
    if cv2.waitKey(1) & 0xff == ord('q'):              
        break
 
# Release the capture once all the processing is done.
video_capture.release()                                
cv2.destroyAllWindows()


###############################################################################
######################## For Image ############################################



import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#find path of xml file containing haarcascade file
cascPathface = ("haarcascade_frontalface_default.xml")
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
#Find path to the image you want to detect face and pass it here
image = cv2.imread("1.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#convert image to Greyscale for haarcascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
 
# the facial embeddings for face in input
encodings = face_recognition.face_encodings(rgb)
names = []
# loop over the facial embeddings incase
# we have multiple embeddings for multiple fcaes
for encoding in encodings:
    #Compare encodings with encodings in data["encodings"]
    #Matches contain array with boolean values and True for the embeddings it matches closely
    #and False for rest
    matches = face_recognition.compare_faces(data["encodings"],
    encoding)
    #set name =inknown if no encoding matches
    name = "Unknown"
    # check to see if we have found a match
    if True in matches:
        #Find positions at which we get True and store them
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            #Check the names at respective indexes we stored in matchedIdxs
            name = data["names"][i]
            #increase count for the name we got
            counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", image)
    cv2.waitKey(0)


