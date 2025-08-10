"""
Калибровка камеры с использованием шахматной доски.
Использует OpenCV для нахождения углов шахматной доски и калибровки камеры.
"""

import numpy as np
import cv2 as cv
import glob
import os
import json
import argparse


def cam_calibration(args):
    """
    Функция для калибровки камеры с использованием шахматной доски.
    :cam: str - имя камеры ('cam_left' или 'cam_right')
    :image_dir: str - директория с изображениями для калибровки
    :output_dir: str - директория для сохранения калибровочных данных
    :cell_size: float - размер клетки шахматной доски в метрах
    :pattern_size: tuple - размер шахматной доски (количество внутренних углов)
    :show_images: bool - показывать ли изображения с углами шахматной доски
    Выходные данные: калибровочные данные сохраняются в формате JSON в папке 'calib'.
    Структура папок:
    ├──image_dir
    │    ├── cam_left
    │    │   ├── image1.png
    │    │   ├── image2.png
    │    │   └── ...
    │    └── cam_right
    │        ├── image1.png
    │        ├── image2.png
    │        └── ...
    └──output_dir
        ├── calibration_cam_left.json
        └── calibration_cam_right.json
    """

    # Критерий остановки
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cam = args.cam

    # Подготовка объектных точек
    cell_size = args.cell_size  # Размер клетки шахматной доски в метрах
    pattern_size = (
        args.pattern_size
    )  # Размер шахматной доски (количество внутренних углов)

    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * cell_size
    )

    # Массивы для хранения объектных точек и точек изображения со всех изображений
    objpoints = []  # 3D точки в пространстве реального мира
    imgpoints = []  # 2D точки в плоскости изображения

    # Путь к папке с изображениями
    image_dir = f"{args.image_dir}/{cam}"
    images = glob.glob(os.path.join(image_dir, "*.png"))

    if not images:
        print("Ошибка: не найдено изображений в формате PNG в директории.")
    else:
        for fname in images:
            img = cv.imread(fname)
            if img is None:
                print(f"Ошибка: не удалось загрузить изображение {fname}.")
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Находим углы шахматной доски
            ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

            # Если углы найдены, добавляем их в массивы
            if ret:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Рисуем углы на изображении
                if args.show_images:
                    cv.drawChessboardCorners(img, pattern_size, corners2, ret)
                    cv.imshow("img", img)
                    cv.waitKey(500)

        cv.destroyAllWindows()

    # Калибровка камеры
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Выводим результаты калибровки
    print(f"RMS error: {ret}")

    # Сохранение калибровочных данных
    calib_data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
    calib_file = os.path.join(args.output_dir, f"calibration_{cam}.json")
    with open(calib_file, "w") as f:
        json.dump(calib_data, f)
    print(f"Калибровочные данные сохранены в {calib_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cam",
        type=str,
        required=True,
        help="Камера для калибровки (cam_left or cam_right)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="screenshot",
        help="Директория с изображениями для калибровки",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="calib",
        help="Директория для сохранения калибровочных данных",
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        default=0.025,
        help="Размер ячейки шахматной доски в метрах",
    )

    parser.add_argument(
        "--pattern_size",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(9, 6),
        help='Размер шахматной доски (количество внутренних углов), например "9,6"',
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="Показать изображения с углами шахматной доски",
    )
    args = parser.parse_args()

    cam_calibration(args)
