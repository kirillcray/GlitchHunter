"""
Калибровка стереокамеры с использованием шахматной доски.
Использует OpenCV для нахождения углов шахматной доски и калибровки камеры.
Необходима предварительная калибровка отдельных камер.
Сохраняет калибровочные данные в формате JSON.
"""

import numpy as np
import cv2 as cv
import glob
import os
import json
import argparse


def stereo_calibration(args):
    """
    Функция для стереокалибровки камер.
    :image_dir_L: str - директория с изображениями левой камеры
    :image_dir_R: str - директория с изображениями правой камеры
    :output_dir: str - директория для сохранения калибровочных данных
    :cell_size: float - размер клетки шахматной доски в метрах
    :pattern_size: tuple - размер шахматной доски (количество внутренних углов)
    :show_images: bool - показывать ли изображения с углами шахматной доски
    Выходные данные: калибровочные данные сохраняются в формате JSON в папке 'calib'.
    Cтруктура папок:
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
        ├── calibration_cam_right.json
        └── stereo_calibration_data.json
    """

    # Критерий остановки
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Подготовка объектных точек
    cell_size = args.cell_size  # Размер клетки шахматной доски в метрах
    pattern_size = (
        args.pattern_size
    )  # Размер шахматной доски (количество внутренних углов)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * cell_size
    )

    # Массивы для хранения объектных точек и точек изображения
    objpoints = []  # 3D точки в пространстве реального мира
    imgpointsL = []  # 2D точки в плоскости изображения для левой камеры
    imgpointsR = []  # 2D точки в плоскости изображения для правой камеры

    # Загружаем изображения
    imagesL = glob.glob(os.path.join(args.image_dir_L, "*.png"))
    imagesR = glob.glob(os.path.join(args.image_dir_R, "*.png"))

    if not imagesL or not imagesR:
        print("Ошибка: не найдено изображений в формате PNG в директориях.")
        return

    # Склеиваем изображения по парам и калибруем
    for fnameL, fnameR in zip(imagesL, imagesR):
        imgL = cv.imread(fnameL)
        imgR = cv.imread(fnameR)

        if imgL is None or imgR is None:
            print(f"Ошибка: не удалось загрузить изображения {fnameL} и {fnameR}.")
            continue

        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Находим углы шахматной доски
        retL, cornersL = cv.findChessboardCorners(grayL, pattern_size, None)
        retR, cornersR = cv.findChessboardCorners(grayR, pattern_size, None)

        # Если углы найдены, добавляем их в массивы
        if retL and retR:
            objpoints.append(objp)

            corners2L = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(corners2L)

            corners2R = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(corners2R)

            # Отображаем углы (если включён флаг --show_images)
            if args.show_images:
                cv.drawChessboardCorners(imgL, pattern_size, corners2L, retL)
                cv.drawChessboardCorners(imgR, pattern_size, corners2R, retR)
                cv.imshow("imgL", imgL)
                cv.imshow("imgR", imgR)
                cv.waitKey(500)

    cv.destroyAllWindows()

    # Загружаем калибровочные данные камер
    with open(os.path.join(args.output_dir, "calibration_cam_left.json"), "r") as f:
        calib_data_L = json.load(f)
    with open(os.path.join(args.output_dir, "calibration_cam_right.json"), "r") as f:
        calib_data_R = json.load(f)

    cameraMatrixL = np.array(calib_data_L["camera_matrix"])
    distCoeffsL = np.array(calib_data_L["dist_coeff"])
    cameraMatrixR = np.array(calib_data_R["camera_matrix"])
    distCoeffsR = np.array(calib_data_R["dist_coeff"])

    # Стереокалибровка
    ret, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = (
        cv.stereoCalibrate(
            objpoints,
            imgpointsL,
            imgpointsR,
            cameraMatrixL,
            distCoeffsL,
            cameraMatrixR,
            distCoeffsR,
            grayL.shape[::-1],
            criteria=criteria,
            # flags=cv.CALIB_FIX_INTRINSIC
        )
    )

    # Выводим результаты калибровки
    print(f"RMS error: {ret}")

    # Сохраняем стереокалибровочные данные
    stereo_calib_data = {
        "cameraMatrixL": cameraMatrixL.tolist(),
        "distCoeffsL": distCoeffsL.tolist(),
        "cameraMatrixR": cameraMatrixR.tolist(),
        "distCoeffsR": distCoeffsR.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
    }

    stereo_calib_file = os.path.join(args.output_dir, "stereo_calibration_data.json")
    with open(stereo_calib_file, "w") as f:
        json.dump(stereo_calib_data, f)
    print(f"Стереокалибровочные данные сохранены в {stereo_calib_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir_L",
        type=str,
        default="screenshot/cam_left",
        help="Путь к изображениям левой камеры",
    )
    parser.add_argument(
        "--image_dir_R",
        type=str,
        default="screenshot/cam_right",
        help="Путь к изображениям правой камеры",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="calib",
        help="Директория для сохранения калибровочных данных",
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="Показать изображения с углами шахматной доски",
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
    args = parser.parse_args()
    stereo_calibration(args)
