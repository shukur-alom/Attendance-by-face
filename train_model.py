import cv2,os,pickle
import face_recognition

encodin_list , class_name = [],[]

for i in os.listdir('data'):

    img = cv2.cvtColor(cv2.imread(f"data/{i}"), cv2.COLOR_BGR2RGB)
    encodin_list.append( face_recognition.face_encodings(img)[0] )
    class_name.append(i.split()[0])

print(class_name)

pickle.dump( encodin_list, open( "encodin list.p", "wb" ) )
pickle.dump( class_name, open( "class name.p", "wb" ) )

print("Training Done........")