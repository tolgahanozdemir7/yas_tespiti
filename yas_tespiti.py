# DeepFace Kutuphanesi ile yas_tespiti
import cv2
from deepface import DeepFace

cap=cv2.VideoCapture(0)
ret,frame=cap.read()
face_cascade=cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

cap.release()
cv2.imwrite("res.jpg",frame)
img=cv2.imread("res.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

faces=face_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,256,0),3)
    cv2.imshow("img",img)

resp=DeepFace.analyze(img,["age"])
print(resp)
age_result=resp[0]["age"]
print(age_result)

if age_result<20:
    print("genc gorunuyorsun")
elif age_result<40:
    print("orta yas gorunuyorsun")
elif age_result<60:
    print("olgun gorunuyorsun")
else :
    print("yasli gorunuyorsun")
    
cv2.waitKey(0)
cv2.destroyAllWindows()



