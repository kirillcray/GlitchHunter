import os
import numpy as np
import cv2  # OpenCV для CLAHE


def normalize(
    coord_min=[300, 290],
    coord_max=[810, 910],
    input_path=r"depth_maps\npy",
    output_path=r"cropped_images",
):
    os.makedirs(output_path, exist_ok=True)  # Создаем папку, если она не существует

    images = []  # Список для хранения обработанных изображений
    filenames = []  # Список для хранения имен файлов

    for filename in os.listdir(input_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_path, filename)
            image = np.load(file_path)
            print(f"Обработка файла: {filename}")
            new_image = image[coord_min[0] : coord_max[0], coord_min[1] : coord_max[1]]
            new_image_uint8 = (
                (new_image - new_image.min())
                / (new_image.max() - new_image.min())
                * 255
            ).astype(np.uint8)

            images.append(new_image_uint8)
            filenames.append(filename)

            # Сохранение обработанного изображения
            save_path = os.path.join(output_path, filename.replace(".npy", ".png"))
            cv2.imwrite(save_path, new_image_uint8)
            print(f"Сохранено: {save_path}")


if __name__ == "__main__":
    normalize()
