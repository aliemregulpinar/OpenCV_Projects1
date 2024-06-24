import cv2

# Cascade dosyalarını yükleme
yuzCascade = cv2.CascadeClassifier('Cascades/haarcascades/haarcascade_frontalface_default.xml')
gulCascade = cv2.CascadeClassifier('Cascades/haarcascades/haarcascade_smile.xml')

# Kamera açma ve ayarları
kamera = cv2.VideoCapture(0)
kamera.set(3, 1280)  # Genişlik
kamera.set(4, 720)  # Yükseklik

# Video kaydedici için değişkenler
dosyaadi = "gulusKayit.mp4"
kaydedici = None

while True:
    # Kameradan kare okuma
    ret, kare = kamera.read()
    if not ret:
        break

    # Gri tonlamalı hale çevirme
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    # Yüz algılama
    yuzler = yuzCascade.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in yuzler:
        # Yüz için dikdörtgen çizme
        cv2.rectangle(kare, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gri_kutu = gri[y:y + h, x:x + w]
        renkli_kutu = kare[y:y + h, x:x + w]

        # Gülüş algılama
        gulusler = gulCascade.detectMultiScale(gri_kutu, scaleFactor=1.5, minNeighbors=18, minSize=(30, 30))

        for (sx, sy, sw, sh) in gulusler:
            # Gülüş için dikdörtgen çizme
            cv2.rectangle(renkli_kutu, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    # Kareyi gösterme
    cv2.imshow('kare', kare)

    # Video kaydedici başlatma
    if kaydedici is None and dosyaadi is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        kaydedici = cv2.VideoWriter(dosyaadi, fourcc, 24.0, (kare.shape[1], kare.shape[0]), True)

    # Videoyu kaydetme
    if kaydedici is not None:
        kaydedici.write(kare)

    # Çıkış için tuşa basma
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord('q'):
        break

# Kamerayı serbest bırakma ve pencereleri kapatma
kamera.release()
if kaydedici:
    kaydedici.release()
cv2.destroyAllWindows()