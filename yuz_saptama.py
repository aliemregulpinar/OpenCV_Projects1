import cv2

yuzCascade = cv2.CascadeClassifier('Cascades/haarcascades/haarcascade_frontalface_default.xml')
kamera = cv2.VideoCapture(0)
while True:
    _, kare = kamera.read()
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = yuzCascade.detectMultiScale(gri, scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(20, 20))
    for (x, y, w, h) in yuzler:
        cv2.rectangle(kare, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('kare', kare)
        # q tuşuna basınca kapat.
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == ord('q'):
            break

cv2.destroyAllWindows()
