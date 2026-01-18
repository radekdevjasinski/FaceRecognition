import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH = 'emotion_cnn_model_v2.h5'
IMAGE_SIZE = 48
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FRAME_SKIP = 60

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face recognition")
        self.root.geometry("450x350")
        self.root.resizable(False, False)
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.display_text = "Inicjalizacja..."

        self._setup_ui()
        self.root.after(200, self._load_resources)

    def _setup_ui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(main_frame, text="Wykrywanie emocji", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        self.status_var = tk.StringVar()
        self.status_var.set("Inicjalizacja...")
        self.status_label = tk.Label(main_frame, textvariable=self.status_var, fg="gray", font=("Arial", 10))
        self.status_label.pack(pady=(0, 20))

        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack()

        self.btn_camera = tk.Button(buttons_frame, text="Uruchom Kamerę (na żywo)",
                                    command=self.start_camera_action,
                                    width=25, height=2, font=("Arial", 11), state=tk.DISABLED, bg="#e1e1e1")
        self.btn_camera.pack(pady=10)

        self.btn_image = tk.Button(buttons_frame, text="Wczytaj Zdjęcie z pliku...",
                                   command=self.load_image_action,
                                   width=25, height=2, font=("Arial", 11), state=tk.DISABLED, bg="#e1e1e1")
        self.btn_image.pack(pady=10)

        btn_quit = tk.Button(main_frame, text="Zakończ", command=self.root.quit, font=("Arial", 10))
        btn_quit.pack(side=tk.BOTTOM, pady=(20, 0))

    # ladowanie modelu
    def _load_resources(self):
        self.status_var.set("Ładowanie modelu.")
        self.status_label.config(fg="blue")
        self.root.update_idletasks()
        try:
            # tensorflow load_model()
            self.model = load_model(MODEL_PATH)
            self.status_var.set("Model gotowy.")
            self.status_label.config(fg="green")
            # odblokuj przyciski akcji
            self.btn_camera.config(state=tk.NORMAL, bg="#d0f0c0")
            self.btn_image.config(state=tk.NORMAL, bg="#d0f0c0")
        except Exception as e:
            err_msg = f"Nie udało się załadować pliku modelu: {MODEL_PATH}\n\nSzczegóły: {e}"
            self.status_var.set("Sprawdź plik modelu.")
            self.status_label.config(fg="red")
            messagebox.showerror("Błąd inicjalizacji", err_msg)

    def preprocess_face(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img

    # wykonanie predycji emocji poprzez wytrenowany model
    def predict_emotion(self, face_roi):
        # dostosowanie zdjecia do modelu
        processed_face = self.preprocess_face(face_roi)

        # predykcja emocji
        prediction = self.model.predict(processed_face, verbose=0)

        # wypisanie emocji z ktora posiada najwieksza pewnosc
        emotion_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return EMOTION_LABELS[emotion_idx], confidence

    def start_camera_action(self):
        if not self.model: return

        self.status_var.set("Kamera aktywna")
        self.root.update()

        # uzycie kamery poprzez opencv
        cap = cv2.VideoCapture(0)
        self.display_text = "Analizuję..."
        frame_count = 0

        if not cap.isOpened():
            messagebox.showerror("Błąd", "Nie można otworzyć kamery.")
            self.status_var.set("Model gotowy.")
            return

        # dla kazdej klatki
        while True:
            # pobierz klatke kamery
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1

            # wykryj twarze z obrazu kamery
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            #dla kazdej twarzy wykryj emocje i narysuj ramke
            for (x, y, w, h) in faces:
                # wykonuj czynnosc raz na FRAME_SKIP klatek
                if frame_count % FRAME_SKIP == 0:
                    face_roi = frame[y:y + h, x:x + w]
                    current_emotion, confidence = self.predict_emotion(face_roi)
                    self.display_text = f"{current_emotion.upper()} ({confidence:.1f}%)"

                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 25), (x + w, y), color, -1)
                cv2.putText(frame, self.display_text, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow('Podglad Kamery', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

            try:
                self.root.winfo_exists()
            except tk.TclError:
                break

        cap.release()
        cv2.destroyAllWindows()

        try:
            self.status_var.set("Model gotowy do pracy.")
            self.status_label.config(fg="green")
        except tk.TclError:
            pass

    def load_image_action(self):
        if not self.model: return

        # wybranie zdjecia z folderu
        file_path = filedialog.askopenfilename(
            title="Wybierz plik obrazu",
            filetypes=[("Pliki obrazów", "*.jpg *.jpeg *.png *.bmp"), ("Wszystkie pliki", "*.*")]
        )

        if not file_path: return

        self.status_var.set(f"Przetwarzanie: {os.path.basename(file_path)}")
        self.root.update()

        # odczytanie zdjecia przez opencv
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Błąd", "Nie udało się wczytać wybranego pliku jako obrazu.")
            self.status_var.set("Model gotowy do pracy.")
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # wykrycie twarzy na zdjeciu przez opencv
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            messagebox.showinfo("Info", "Nie wykryto żadnej twarzy na zdjęciu.")
        else:
            # wykonaj wykrycie emocji dla kazdej twarzy
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                emotion_text, confidence = self.predict_emotion(face_roi)
                display_text = f"{emotion_text} ({confidence:.1f}%)"

                # stworzenie ramki wokol twarzy z etykieta emocji
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # wyswietlenie zdjecia z rozpoznana emocja
            cv2.imshow('Wynik analizy zdjecia', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.status_var.set("Model gotowy do pracy.")
        self.status_label.config(fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()