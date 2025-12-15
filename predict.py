import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # nowy import
from tensorflow.keras.models import load_model

# ustawienia
IMAGE_SIZE = 48
MODEL_SAVE_PATH = 'emotion_cnn_model.h5'
PREDICT_DATA_DIR = './predict'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_images_for_prediction(data_dir, image_size):
    # wczytuje i przetwarza obrazy z folderu do predykcji
    data_list = []
    raw_images_list = []
    filenames_list = []
    required_shape = (image_size, image_size)

    processed_count = 0
    discarded_count = 0

    print(f"ladowanie obrazow do predykcji z: {data_dir}")

    # sprawdz, czy folder istnieje
    if not os.path.isdir(data_dir):
        print(f"blad: folder '{data_dir}' nie istnieje.")
        return None, None, None, None

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            image_path = os.path.join(data_dir, filename)

            # wczytanie obrazu w skali szarosci (IMREAD_GRAYSCALE)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                discarded_count += 1
                continue

            if image.shape != required_shape:
                image_resized = cv2.resize(image, required_shape, interpolation=cv2.INTER_LINEAR)
                print(f"skalowanie {filename} z {image.shape} do {required_shape}")
            else:
                image_resized = image

            if image_resized is None:
                discarded_count += 1
                continue


            raw_images_list.append(image_resized)
            data_list.append(image_resized)
            filenames_list.append(filename)
            processed_count += 1

    if not data_list:
        print("nie znaleziono obrazow do predykcji.")
        return None, None, None, None

    # konwersja do numpy i normalizacja
    X_pred = np.array(data_list, dtype='float32')
    X_pred = X_pred.reshape(-1, image_size, image_size, 1)
    X_pred /= 255.0

    print(f"zaladowano {len(data_list)} obrazow do predykcji.")
    print(f"pominieto {discarded_count} plikow.")

    return X_pred, raw_images_list, filenames_list, len(EMOTION_LABELS)


def predict_and_display_results(model, X_pred, raw_images, filenames, emotion_labels):
    # dokonuje predykcji, wyswietla wyniki tekstowo i graficznie

    print("\nrozpoczecie predykcji...")

    predictions = model.predict(X_pred)
    predicted_classes_index = np.argmax(predictions, axis=1)

    print("\n wyniki predykcji ")

    num_images = len(filenames)
    # przygotowanie do wyswietlania (maksymalnie 4 obrazy w rzedzie)
    cols = 4
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(cols * 4, rows * 4))  # dopasowanie rozmiaru okna do liczby obrazow

    for i in range(num_images):
        filename = filenames[i]

        # dane najlepszej predykcji
        predicted_index = predicted_classes_index[i]
        predicted_emotion = emotion_labels[predicted_index]
        confidence = predictions[i][predicted_index] * 100

        # wyswietlanie tekstowe najlepszej predykcji
        print(f"\n[{filename}] -> Emocja: **{predicted_emotion}** (pewnosc: {confidence:.2f}%)")

        # pelny rozklad prawdopodobienstw
        print("    Pelny rozklad:")

        # tworzymy liste par (procent, etykieta)
        emotion_probabilities = []
        for j, prob in enumerate(predictions[i]):
            emotion_probabilities.append((prob * 100, emotion_labels[j]))

        # sortowanie, aby najpierw byly najwyzsze procenty (opcjonalnie, ale poprawia czytelnosc)
        emotion_probabilities.sort(key=lambda item: item[0], reverse=True)

        for prob, label in emotion_probabilities:
            # formatowanie i wyswietlanie wszystkich emocji
            print(f"      - {label:<10}: {prob:.2f}%")

        # konfiguracja wykresu (subplot)
        plt.subplot(rows, cols, i + 1)

        # wyswietlanie obrazu
        plt.imshow(raw_images[i], cmap='gray')
        plt.title(f"{predicted_emotion}\n({confidence:.2f}%)", fontsize=12)
        plt.xlabel(filename, fontsize=10)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()  # pokazanie wszystkich wykresow


# glowna logika predykcji
if __name__ == "__main__":

    # 1. wczytanie modelu
    try:
        print(f"wczytywanie modelu z: {MODEL_SAVE_PATH}")
        final_model = load_model(MODEL_SAVE_PATH)
        print("model zaladowany pomyslnie.")
    except Exception as e:
        print(f"blad krytyczny: nie mozna wczytac modelu z {MODEL_SAVE_PATH}. {e}")
        exit()

    # 2. wczytanie i przetworzenie danych
    X_pred, raw_images, filenames, num_classes = load_images_for_prediction(PREDICT_DATA_DIR, IMAGE_SIZE)

    if X_pred is None:
        print("brak danych do predykcji. upewnij sie, ze folder istnieje i zawiera pliki.")
    else:
        # 3. wykonanie predykcji i wyswietlenie wynikow
        predict_and_display_results(final_model, X_pred, raw_images, filenames, EMOTION_LABELS)

    print("\npredykcja zakonczona.")