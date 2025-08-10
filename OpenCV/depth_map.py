"""
Модуль для построения карты глубины с использованием изображений стереокамер.
Включает функции для вычисления карты глубины и визуализации эпиполярных линий.
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from stereo_rectify import load_pics, image_calibration, draw_epilines


def disparity_map(imgL, imgR):
    """
    Строит карту глубины на основе изображений стереокамер.

    :param imgL: Изображение левой камеры.
    :param imgR: Изображение правой камеры.
    Выводит карту глубины для различных параметров.
    """
    # Преобразуем изображения в оттенки серого для построения карты глубины
    imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    # Подбор параметров для построения карты глубины
    numDisparities_values = [256]
    blockSize_values = [7, 9, 11, 13, 17, 25, 27, 29]
    # Построение карты глубины
    h, w = imgR_gray.shape[:2]
    for numDisparities in numDisparities_values:
        if numDisparities % 16 != 0 or numDisparities >= w:
            continue
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"numDisparities={numDisparities}")
        for i, blockSize in enumerate(blockSize_values):
            stereo = cv.StereoSGBM_create(
                minDisparity=0,  # Минимальное значение диспаратности
                numDisparities=numDisparities,  # Количество диспаратностей
                blockSize=blockSize,  # Размер блока
                uniquenessRatio=10,  # Минимальное отношение между наименьшим и наибольшим значением диспаратности
                speckleWindowSize=0,  # Размер области сглаживания
                speckleRange=2,  # Максимальная разница в диспаратности
                disp12MaxDiff=10,  # Максимальная разница в диспаратности между соседними пикселями
                P1=8 * 3 * blockSize**2,
                P2=32 * 3 * blockSize**2,
            )
            disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32)
            disparity = cv.normalize(
                disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX
            )
            disparity = np.uint8(disparity)
            plt.subplot(2, 4, i + 1)
            plt.imshow(disparity, "gray")
            plt.title(f"blockSize={blockSize}")
            plt.axis("off")
        plt.show()


if __name__ == "__main__":

    imgL, imgR = load_pics(name="17-44-36.png")
    imgL, imgR = image_calibration(imgL=imgL, imgR=imgR)
    disparity_map(imgL=imgL, imgR=imgR)
    draw_epilines(imgL, imgR)
    cv.destroyAllWindows()
