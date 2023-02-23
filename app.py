from flask import Flask,render_template,Response,request,flash,get_flashed_messages
import face_recognition
import csv
import cv2
import os
import os.path
import time
import numpy as np
from PIL import Image
from threading import Thread
app = Flask(__name__)
app.secret_key = 'random string'
cap = cv2.VideoCapture(0)
@app.route('/register',methods=['GET','POST'])
def index():
    if request.method=='POST': 
        empName = request.form['empName']
        empId = request.form['empId']
        success, frame = cap.read()
        print(os.path.abspath("TrainingImage"))
        cv2.imwrite(f"{os.path.abspath('TrainingImage')}{os.sep}{empName}-{empId}.jpg",frame)
    return render_template('index.html', page_title="Registration")
@app.route('/attendance',methods=['GET','POST'])
def attendance():
    # if request.method=='POST': 
    #     empName = request.form['empName']
    #     empId = request.form['empId']
    #     success, frame = cap.read()
    #     print(os.path.abspath("TrainingImage"))
    #     cv2.imwrite(f"{os.path.abspath('TrainingImage')}{os.sep}{empName}-{empId}.jpg",frame)
    return render_template('attendance.html', page_title="Attendance")
@app.route('/visitor',methods=['GET','POST'])
def visitor():
    lenlstImgs = os.listdir('TrainingImage')
    print(lenlstImgs)
    lstImagesEncodings=[]
    for i in range(len(lenlstImgs)):
        img = cv2.imread(fr".{os.sep}TrainingImage{os.sep}{lenlstImgs[i]}")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lstImagesEncodings.append(face_recognition.face_encodings(rgb_img)[0])
    success, frame = cap.read()
    flash("Result: Recognizing...")
    img_encoding = face_recognition.face_encodings(frame)[0]
    arrOfBoolValues = face_recognition.compare_faces(lstImagesEncodings, img_encoding)
    print(arrOfBoolValues)
    idxOfTrue=-1
    if True in arrOfBoolValues:
        idxOfTrue = arrOfBoolValues.index(True)
        print("Result: ", lenlstImgs[idxOfTrue][0:-4])
        flash(f"Result: {lenlstImgs[idxOfTrue][0:-4]}")
    else:
        flash("Face Not Found")
    return render_template('visitor.html', page_title="Visitor")
def gen_frames():  
    while True:
        success, frame = cap.read()  # read the camera frame
        face_cascade = cv2.CascadeClassifier(r'.\haarcascade_frontalface_default.xml') 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.25, 4)
        if not success:
            break
        else:
            
            for (x,y,w,h) in faces: 
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
             
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')                
# -------------- image labesl ------------------------
if __name__ == "__main__":
    # app.run(debug=True,host="192.168.0.104")
    app.run(debug=True)
# def getImagesAndLabels(path):
#     # get the path of all the files in the folder
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     # print(imagePaths)

#     # create empth face list
#     faces = []
#     # create empty ID list
#     Ids = []
#     # now looping through all the image paths and loading the Ids and the images
#     for imagePath in imagePaths:
#         # loading the image and converting it to gray scale
#         pilImage = Image.open(imagePath).convert('L')
#         # Now we are converting the PIL image into numpy array
#         imageNp = np.array(pilImage, 'uint8')
#         # getting the Id from the image
#         Id = int(os.path.split(imagePath)[1].split("-")[1])
#         # extract the face from the training image sample
#         faces.append(imageNp)
#         Ids.append(Id)
#     return faces, Ids

# # Optional, adds a counter for images trained (You can remove it)
# def counter_img(path):
#     imgcounter = 1
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     for imagePath in imagePaths:
#         print(str(imgcounter) + " Images Trained", end="\r")
#         time.sleep(0.008)
#         imgcounter += 1
# # ----------- train images function ---------------
# def TrainImages():
#     recognizer = cv2.face.LBPHFaceRecognizer.create()
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     detector = cv2.CascadeClassifier(harcascadePath)
#     faces, Id = getImagesAndLabels("TrainingImage")
#     Thread(target = recognizer.train(faces, np.array(Id))).start()
#     # Below line is optional for a visual counter effect
#     Thread(target = counter_img("TrainingImage")).start()
#     recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
#     print("All Images")








# TrainImages()

# face_cascade = cv2.CascadeClassifier(r'.\haarcascade_frontalface_default.xml') 
# eye_cascade = cv2.CascadeClassifier(r'.\haarcascade_eye.xml') 

# cap = cv2.VideoCapture(0)
# # Id = input("Enter Your Id: ")
# # name = input("Enter Your Name: ")
# sampleNum=0
# while True:  
#     ret, img = cap.read()  
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#     faces = face_cascade.detectMultiScale(gray, 1.25, 4) 
  
#     for (x,y,w,h) in faces: 
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
#         sampleNum+=1
#         rec_gray = gray[y:y+h, x:x+w] 
#         rec_color = img[y:y+h, x:x+w] 
#         cv2.imwrite(f"TrainingImage{os.sep}{name}-{Id}-{str(sampleNum)}.jpg", gray[y:y+h, x:x+w])
#         print(f"Capturing Images {sampleNum}")

#     cv2.imshow('Face Recognition',img) 
  
#     k = cv2.waitKey(30) & 0xff
#     if k == 27: 
#         break
#     elif sampleNum > 100:
#         break
  
# cap.release() 
# cv2.destroyAllWindows()