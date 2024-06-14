# Gerekli kütüphaneleri ve modülleri import etme
import cv2
from keras._tf_keras.keras.models import model_from_json
import numpy as np

# JSON dosyasını okuyup modeli yükleme
json_file = open("facialemotionmodel.json", "r")  # Modelin JSON dosyasını aç
model_json = json_file.read()  # JSON dosyasını oku
json_file.close()  # Dosyayı kapat
model = model_from_json(model_json)  # JSON'dan modeli yükle
model.load_weights("facialemotionmodel.h5")  # Model ağırlıklarını yükle

# Haarcascade yüz tanıma dosyasını yükleme
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Haarcascade dosyasının yolu
face_cascade = cv2.CascadeClassifier(haar_file)  # Haarcascade sınıflandırıcısını yükle

# Özellik çıkarma fonksiyonu
def extract_features(image):
    feature = np.array(image)  # Görüntüyü numpy array'e çevir
    feature = feature.reshape(1, 48, 48, 1)  # Görüntüyü modelin beklediği şekle dönüştür
    return feature / 255.0  # Görüntüyü normalize et

# Webcam'den görüntü alma
webcam = cv2.VideoCapture(0)  # Webcam'den görüntü yakalamak için VideoCapture nesnesi oluştur
# Modelin tahmin ettiği sınıfları ve etiketleri belirle
labels = {0: 'saldirgan uyarisi', 1: 'saldirgan degil', 2: 'saldirgan degil', 3: 'saldirgan degil', 4: 'saldirgan degil', 5: 'saldirgan degil', 6: 'saldirgan degil'}

while True:
    ret, im = webcam.read()  # Webcam'den bir kare oku
    if not ret:  # Eğer kare okunamazsa
        print("Webcam'den görüntü alınamadı!")  # Hata mesajı yazdır
        break  # Döngüden çık

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tonlamaya çevir
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Yüzleri tespit et

    for (p, q, r, s) in faces:  # Her yüz için döngü
        image = gray[q:q+s, p:p+r]  # Yüz bölgesini kırp
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Yüz etrafına dikdörtgen çiz
        image = cv2.resize(image, (48, 48))  # Görüntüyü 48x48 boyutlarına yeniden boyutlandır
        img = extract_features(image)  # Görüntüden özellik çıkar
        pred = model.predict(img)  # Model ile tahmin yap
        prediction_label = labels[pred.argmax()]  # Tahmin edilen etiketi al
        cv2.putText(im, prediction_label, (p, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)  # Tahmini görüntüye yaz

    cv2.imshow("Output", im)  # Sonucu göster

    # Kullanıcının 'q' tuşuna basması durumunda çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()  # Webcam'i serbest bırak
cv2.destroyAllWindows()  # Tüm pencereleri kapat
