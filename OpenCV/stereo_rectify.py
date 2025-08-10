"""
Модуль для работы с изображениями стереокамер, включая загрузку изображений,
выравнивание с использованием калибровочных данных и визуализацию эпиполярных линий.
"""

import glob
import os
import cv2 as cv
import json
import numpy as np
from matplotlib import pyplot as plt


def load_pics(path_L="screenshot/left", path_R="screenshot/right", name=None):
    """
    Загружает изображения с левой и правой камеры.

    :param path_L: Путь к папке с изображениями левой камеры.
    :param path_R: Путь к папке с изображениями правой камеры.
    :param name: Имя файла изображения. Если не указано, используется последнее созданное изображение.
    :return: Кортеж из двух изображений (imgL, imgR).
    """
    if name is None:
        name = max(glob.glob(f"{path_L}/*"), key=os.path.getctime).split(os.sep)[-1]

    imgL = cv.imread(f"{path_L}/{name}", cv.IMREAD_COLOR)
    imgR = cv.imread(f"{path_R}/{name}", cv.IMREAD_COLOR)

    if imgL is None or imgR is None:
        print("Ошибка: не удалось загрузить одно или оба изображения.")
        exit()
    return imgL, imgR


def image_calibration(
    imgL, imgR, stereo_calib_file="OpenCV/calib/stereo_calibration_data.json"
):
    """
    Выравнивает изображения с использованием калибровочных данных.

    :param imgL: Изображение левой камеры.
    :param imgR: Изображение правой камеры.
    :param stereo_calib_file: Путь к файлу с калибровочными данными.
    :return: Кортеж из ректифицированных изображений (rectified_imgL, rectified_imgR).
    """
    with open(stereo_calib_file, "r") as f:
        stereo_calib_data = json.load(f)

    cameraMatrixL = np.array(stereo_calib_data["cameraMatrixL"])
    distCoeffsL = np.array(stereo_calib_data["distCoeffsL"])
    cameraMatrixR = np.array(stereo_calib_data["cameraMatrixR"])
    distCoeffsR = np.array(stereo_calib_data["distCoeffsR"])
    R = np.array(stereo_calib_data["R"])
    T = np.array(stereo_calib_data["T"])

    # Ректификация изображений
    h, w = imgR.shape[:2]
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        cameraMatrixL,
        distCoeffsL,
        cameraMatrixR,
        distCoeffsR,
        (w, h),
        R,
        T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=1,
    )

    # Вычисляем матрицы преобразования для ректификации
    mapLx, mapLy = cv.initUndistortRectifyMap(
        cameraMatrixL, distCoeffsL, R1, P1, (w, h), cv.CV_32FC1
    )
    mapRx, mapRy = cv.initUndistortRectifyMap(
        cameraMatrixR, distCoeffsR, R2, P2, (w, h), cv.CV_32FC1
    )

    # Применяем преобразование к изображениям
    rectified_imgL = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR)
    rectified_imgR = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR)

    # Находим общую область для обрезки изображений
    x1 = min(roi1[0], roi2[0])
    y1 = min(roi1[1], roi2[1])
    x2 = max(roi1[0] + roi1[2], roi2[0] + roi2[2])
    y2 = max(roi1[1] + roi1[3], roi2[1] + roi2[3])

    # Обрезаем изображения по общей области
    rectified_imgL = rectified_imgL[y1:y2, x1:x2]
    rectified_imgR = rectified_imgR[y1:y2, x1:x2]

    # Проверяем, что оба изображения имеют одинаковые размеры
    h_L, w_L = rectified_imgL.shape[:2]
    h_R, w_R = rectified_imgR.shape[:2]

    # Обрезаем изображения до минимального размера, если они различаются
    if h_L != h_R or w_L != w_R:
        min_h = min(h_L, h_R)
        min_w = min(w_L, w_R)
        rectified_imgL = rectified_imgL[:min_h, :min_w]
        rectified_imgR = rectified_imgR[:min_h, :min_w]

    return rectified_imgL, rectified_imgR


def draw_epilines(imgL, imgR):
    """
    Рисует эпиполярные линии на изображениях.

    :param imgL: Изображение левой камеры.
    :param imgR: Изображение правой камеры.
    Выводит изображение с нарисованными эпиполярными линиями.
    """
    img_concat = cv.hconcat([imgL, imgR])  # Склеиваем изображения
    for i in range(0, imgL.shape[0], 20):  # Рисуем линии через 20 пикселей
        cv.line(img_concat, (0, i), (img_concat.shape[1], i), (255, 0, 0), 1)
    plt.imshow(img_concat, cmap="gray")
    plt.show()
