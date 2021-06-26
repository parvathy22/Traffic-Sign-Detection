### to speech conversion  
##from gtts import gTTS  
##  
##
##from playsound import playsound  
##  
##text_val = 'All the best for your exam.'  
##  
##language = 'en'  
##
##obj = gTTS(text=text_val, lang=language, slow=False)  
##  
###Here we are saving the transformed audio in a mp3 file named  
### exam.mp3  
##obj.save("exam.mp3")  
##  
### Play the exam.mp3 file  
##playsound("exam.mp3")  
import os
os.remove("exam.mp3")
