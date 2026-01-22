import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



# ustawienia
IMAGE_SIZE = 48
MODEL_SAVE_PATH = 'emotion_cnn_model_v2.h5'
PREDICT_DATA_DIR = './predict'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_images_for_prediction(data_dir, image_size):
    data_list = []
    raw_images_list = []
    filenames_list = []
    required_shape = (image_size, image_size)

    if not os.path.isdir(data_dir):
        print(f"blad: folder '{data_dir}' nie istnieje.")
        return None, None, None, None

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            image_path = os.path.join(data_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None: continue

            if image.shape != required_shape:
                image_resized = cv2.resize(image, required_shape, interpolation=cv2.INTER_LINEAR)
            else:
                image_resized = image

            raw_images_list.append(image_resized)
            data_list.append(image_resized)
            filenames_list.append(filename)

    if not data_list:
        return None, None, None, None

    X_pred = np.array(data_list, dtype='float32')
    X_pred = X_pred.reshape(-1, image_size, image_size, 1)
    X_pred /= 255.0

    return X_pred, raw_images_list, filenames_list, len(EMOTION_LABELS)




def predict_and_display_results(model, X_pred, raw_images, filenames, emotion_labels):
    print("\nrozpoczecie predykcji...")
    predictions = model.predict(X_pred)
    predicted_classes_index = np.argmax(predictions, axis=1)

    num_images = len(filenames)
    images_per_plot = 9
    max_plots = 3

    # Pętla generująca maksymalnie 3 osobne wykresy
    for plot_idx in range(max_plots):
        start_idx = plot_idx * images_per_plot
        end_idx = start_idx + images_per_plot

        # Jeśli nie ma więcej obrazów do wyświetlenia, przerwij
        if start_idx >= num_images:
            break

        # Wybierz obrazy do aktualnego okna (maksymalnie 9)
        current_batch_indices = range(start_idx, min(end_idx, num_images))

        plt.figure(figsize=(10, 10))
        plt.suptitle(f"Predykcje - Okno {plot_idx + 1}", fontsize=16)

        for i in current_batch_indices:
            # Oblicz pozycję w siatce 3x3 (1-9)
            subplot_pos = (i % images_per_plot) + 1

            filename = filenames[i]
            predicted_index = predicted_classes_index[i]
            predicted_emotion = emotion_labels[predicted_index]
            confidence = predictions[i][predicted_index] * 100

            # Log konsoli
            print(f"[{filename}] -> {predicted_emotion} ({confidence:.2f}%)")

            # Dodawanie do subplotu
            plt.subplot(3, 3, subplot_pos)
            plt.imshow(raw_images[i], cmap='gray')
            plt.title(f"Pred: {predicted_emotion}\nConf: {confidence:.1f}%", fontsize=10)
            plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()  # Wyświetla aktualne okno (9 obrazów)


if __name__ == "__main__":
    try:
        final_model = load_model(MODEL_SAVE_PATH)
        print("model zaladowany pomyslnie.")
    except Exception as e:
        print(f"blad: {e}")
        exit()

    X_pred, raw_images, filenames, num_classes = load_images_for_prediction(PREDICT_DATA_DIR, IMAGE_SIZE)

    if X_pred is not None:
        predict_and_display_results(final_model, X_pred, raw_images, filenames, EMOTION_LABELS)

    print("\npredykcja zakonczona.")