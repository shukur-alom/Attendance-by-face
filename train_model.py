import cv2,os
from pickle import load,dump
import face_recognition

encodin_list = load( open( "encodin list.p", "rb" ) )
class_name = load( open( "class name.p", "rb" ) )

user_inp = int(input("1. Add one picture and Training\n2. Add more picture and Training\n3. Clear all previous record data\nChoose One : "))

def add_data(img_path):

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    try:encodin_list.append( face_recognition.face_encodings(img)[0] )
    except:print(f"Face not Detacted. remove {img_path.split('/')[-1]}")
    
    nam = img_path.split('/')[-1]
    class_name.append(nam.split()[0])
    
    print(F"Done {nam}")
    

if user_inp == 1:
    
    dic_tory = input("The location of the picture you want to add : ")
    add_data(dic_tory)

    dump( encodin_list, open( "encodin list.p", "wb" ) )
    dump( class_name, open( "class name.p", "wb" ) )


elif user_inp == 2:
    
    dic_tory = input("The folder location of the images you want to add: ")

    for i in os.listdir(dic_tory):
        
        add_data(f"{dic_tory}/{i}")

    dump( encodin_list, open( "encodin list.p", "wb" ) )
    dump( class_name, open( "class name.p", "wb" ) )

elif user_inp == 3:
    dump( [], open( "encodin list.p", "wb" ) )
    dump( [], open( "class name.p", "wb" ) )

print("\nDone........")