from tkinter import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tkinter.filedialog import askopenfile
from gtts import gTTS
import os
from playsound import playsound
from imutils import paths
import numpy as np
import cv2
from PIL import ImageTk, Image
def load():
    global cimg
    x1=[]
    lb.delete(0,'end')
    M2.config(text="")
    path = askopenfile()
    n=path.name
    cimg=cv2.imread(n)
    L2.config(text=n)
    lb.insert(1, "Image Loaded...") 
    print("image loaded")
    img = ImageTk.PhotoImage(Image.open(n))
    canvas.itemconfig(imagecontainer,image=img)  
    cv2.imshow("Selected image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict():
    language = 'en'
    classes = { 0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)', 
                2:'Speed limit (50km/h)', 
                3:'Speed limit (60km/h)', 
                4:'Speed limit (70km/h)', 
                5:'Speed limit (80km/h)', 
                6:'End of speed limit (80km/h)', 
                7:'Speed limit (100km/h)', 
                8:'Speed limit (120km/h)', 
                9:'No passing', 
                10:'No passing vehicle over 3.5 tons', 
                11:'Right-of-way at intersection', 
                12:'Priority road', 
                13:'Yield', 
                14:'Stop', 
                15:'No vehicles', 
                16:'Vehicle > 3.5 tons prohibited', 
                17:'No entry', 
                18:'General caution', 
                19:'Dangerous curve left', 
                20:'Dangerous curve right', 
                21:'Double curve', 
                22:'Bumpy road', 
                23:'Slippery road', 
                24:'Road narrows on the right', 
                25:'Road at work', 
                26:'Traffic signals', 
                27:'Pedestrians', 
                28:'Children crossing', 
                29:'Bicycles crossing', 
                30:'Beware of ice/snow',
                31:'Wild animals crossing', 
                32:'End speed + passing limits', 
                33:'Turn right ahead', 
                34:'Turn left ahead', 
                35:'Ahead only', 
                36:'Go straight or right', 
                37:'Go straight or left', 
                38:'Keep right', 
                39:'Keep left', 
                40:'Roundabout mandatory', 
                41:'End of no passing', 
                42:'End no passing vehicle > 3.5 tons' }
    print(type(classes))
    lb.insert(2, "image Preprocessing...") 
    img = cimg
    img=cv2.resize(img, (32,32))
    img = img_to_array(img)
    lb.insert(3, " Load Model From Disk..") 
    json_file = open('tr_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("tr_model.h5")
    print("Loaded model from disk")
    img = np.expand_dims(img, axis = 0)
    lb.insert(4, "Traffic Sign Prediction...") 
    result = model.predict(img)
    result=np.argmax(result[0])
    lb.insert(5, "Showing Result..") 
    print((result))
    print(classes[result])
    text_val=classes[result]
    M2.config(text=text_val)
    obj = gTTS(text=text_val, lang=language, slow=False)
    lb.insert(6, "Alert!!!...")
    obj.save("alert.mp3")  
    playsound("alert.mp3")
    os.remove("alert.mp3")

    
def home():
    global root,M1,lb,L2,M2,canvas,imagecontainer
    root=Tk()
    root.title("Traffic Prediction")
    root.geometry("1000x1000")
    root.config(bg="#191919")
    T1=Label(root,text="Driver Warning System Using Traffic Sign Detection", bg='#191919', fg='white',font="Helvetica 20")
    T1.pack()
    B1=Button(root,text="Select Your Image", bg="#6c757d",fg='white',height=2,width=22,font=("Arial", 12),command=load)
    B1.place(x=30,y=150)
    L1=Label(root,text="Image Path :", bg='#191919',fg="white",font=("Arial", 10))
    L1.place(x=30,y=215)
    L2=Label(root,text="",bg='#191919', fg="white",font=("Arial", 8))
    L2.place(x=110,y=215)

    canvas = Canvas(root, width=200, height=200,bg="#191919")
    canvas.place(x=30, y=250)
    img = ImageTk.PhotoImage(Image.open(r"C:\Users\user\Pictures\Desktop\Capture.PNG"))
    imagecontainer=canvas.create_image(0, 0, anchor=NW, image=img)

    B3=Button(root,text="Predict",bg="#007bff",fg='white',height=2,width=22,font=("Arial", 12),command=predict)
    B3.place(x=30,y=480)
    M1=Label(root,text="Predicted Traffic Sign :",bg='#191919', fg="white",font=("Arial", 15))
    M1.place(x=15,y=565)
    M2=Label(root,text="",bg="white", fg="red",font=("Arial", 15))
    M2.place(x=220,y=565)
    T3=Label(root,text="Process",bg='#191919', fg='white',font="Helvetica 20")
    T3.place(x=580,y=100)
    lb= Listbox(root, height=25, width=50,)
    lb.place(x=480,y=150)
    root.mainloop()
home()
