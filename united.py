import os
from datetime import datetime
import subprocess
import sys

# Импортируем модули для обработки изображений и работы с камерами
from image_processing_for_anomalib import image_processing
from OpenCV import cam_read


def create_new_dir(base_path='result'):
    """
    Создаёт новую директорию для хранения результатов.
    Папка создаётся с текущей датой, а внутри создаётся подпапка с версией (v1, v2 и т.д.).
    """
    # Получаем сегодняшнюю дату в формате YYYY-MM-DD
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_path, today)

    # Создаём папку с датой, если она не существует
    os.makedirs(date_dir, exist_ok=True)

    # Получаем список подпапок вида v1, v2, v3...
    existing_versions = [
        d for d in os.listdir(date_dir)
        if os.path.isdir(os.path.join(date_dir, d)) and d.startswith("v") and d[1:].isdigit()
    ]

    # Определяем следующий номер версии
    next_version_number = 1
    if existing_versions:
        # Извлекаем номера версий и находим максимальный, чтобы определить следующую версию
        version_numbers = [int(d[1:]) for d in existing_versions]
        next_version_number = max(version_numbers) + 1

    # Формируем имя и путь для новой версии
    version_dir_name = f"v{next_version_number}"
    version_dir_path = os.path.join(date_dir, version_dir_name)

    # Создаем папку для новой версии
    os.makedirs(version_dir_path)
    return version_dir_path


def take_photo(output_path):
    """
    Инициализирует камеры и сохраняет изображения в указанную директорию.
    """
    # Инициализируем левую и правую камеры
    cam_L, cam_R = cam_read.initialize_cameras(cam_port_L=0, cam_port_R=1,
                                               width=1920, height=1080)
    # Обрабатываем поток с камер и сохраняем изображения
    cam_read.process_camera_stream(cam_L, cam_R, output_dir=output_path)


def run_monster(input_left_path, input_right_path, output_path):
    """
    Запускает внешний скрипт MonSter для обработки изображений.
    """
    command = [
        sys.executable, r"MonSter\save_disp.py",  # Путь к скрипту MonSter
        "--left_imgs", input_left_path,          # Путь к левым изображениям
        "--right_imgs", input_right_path,        # Путь к правым изображениям
        "--output_directory", output_path        # Путь для сохранения результатов
    ]
    # Выполняем команду и проверяем успешность выполнения
    subprocess.run(command, check=True)


def preprocess_image(coord_min, coord_max, input_path, output_path):
    """
    Нормализует изображения в указанной директории.
    """
    # Вызываем функцию нормализации из модуля обработки изображений
    image_processing.normalize(coord_min=coord_min, coord_max=coord_max,
                            input_path=input_path, output_path=output_path)


def run_anomalib(input_path, checkpoint_path, output_path):
    """
    Запускает Anomalib для анализа аномалий.
    """
    command = [
        sys.executable, r"anomalib\test_anomaly_detector.py",  # Путь к скрипту Anomalib
        "--input_path", input_path,                           # Путь к входным данным
        "--checkpoint_path", checkpoint_path,                 # Путь к контрольной точке модели
        "--output_path", output_path                          # Путь для сохранения результатов
    ]
    # Выполняем команду и проверяем успешность выполнения
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # Основной рабочий процесс
    base_path = 'result'  # Базовая директория для хранения результатов

    # Создаём новую директорию для текущей сессии
    current_path = create_new_dir(base_path=base_path)
    print(current_path)

    # Снимаем фотографии и сохраняем их в папку rectified_images
    take_photo(output_path=f'{current_path}/rectified_images')

    # Запускаем MonSter для обработки изображений
    run_monster(
        input_left_path=f'{current_path}/rectified_images/left/*.png',
        input_right_path=f'{current_path}/rectified_images/right/*.png',
        output_path=f'{current_path}/MonSter'
    )

    # Нормализуем изображения и сохраняем их в папку cropped_images
    preprocess_image(
        coord_min=[300, 290], coord_max=[810, 910],
        input_path=f"{current_path}/MonSter/npy",
        output_path=f"{current_path}/cropped_images"
    )

    # Запускаем Anomalib для анализа аномалий
    run_anomalib(
        input_path=f'{current_path}/cropped_images',
        checkpoint_path=r'anomalib\results\patchcore\my_board\v3\weights\lightning\model.ckpt',
        output_path=f'{current_path}/anom'
    )