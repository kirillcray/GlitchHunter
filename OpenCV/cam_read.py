"""
Модуль для работы с камерами, включая захват изображений, их сохранение,
выравнивание и построение карты глубины.
"""

import os
import cv2 as cv
import datetime as dt
from .stereo_rectify import image_calibration, draw_epilines


def initialize_cameras(cam_port_L=0, cam_port_R=1, width=1920, height=1080):
    """
    Инициализирует камеры и устанавливает разрешение.

    :param cam_port_L: Порт левой камеры.
    :param cam_port_R: Порт правой камеры.
    :param width: Желаемая ширина кадра.
    :param height: Желаемая высота кадра.
    :return: Объекты VideoCapture для левой и правой камер.
    """
    cam_L = cv.VideoCapture(cam_port_L, cv.CAP_MSMF)
    cam_R = cv.VideoCapture(cam_port_R, cv.CAP_MSMF)

    cam_L.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cam_L.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cam_R.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cam_R.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    return cam_L, cam_R


def process_camera_stream(cam_L, cam_R, output_dir="rectified_images"):
    """
    Обрабатывает поток с камер, отображает изображения, сохраняет их
    и выполняет выравнивание и построение карты глубины.

    :param cam_L: Объект VideoCapture для левой камеры.
    :param cam_R: Объект VideoCapture для правой камеры.
    """
    print("ЭТО ПУТЬ ", output_dir)
    while True:
        # Считывание изображения с камер
        result_1, image_L = cam_L.read()
        result_2, image_R = cam_R.read()
        image_L = cv.rotate(image_L, cv.ROTATE_90_CLOCKWISE)
        image_R = cv.rotate(image_R, cv.ROTATE_90_COUNTERCLOCKWISE)

        # Отображение изображений
        cv.namedWindow("frame_L", cv.WINDOW_NORMAL)
        cv.namedWindow("frame_R", cv.WINDOW_NORMAL)
        cv.resizeWindow("frame_L", 540, 960)
        cv.resizeWindow("frame_R", 540, 960)
        cv.imshow("frame_L", image_L)
        cv.imshow("frame_R", image_R)

        # Ожидание нажатия клавиши
        key = cv.waitKey(10) & 0xFF

        # Обработка нажатия клавиши
        if key in [ord("d"), ord("s")]:
            time = dt.datetime.now().time()
            filename_L = f"{time.hour}-{time.minute}-{time.second}.png"

            # Сохранение изображений
            if key == ord("s"):
                cv.imwrite(f"OpenCV/screenshot/left/{filename_L}", image_L)
                cv.imwrite(f"OpenCV/screenshot/right/{filename_L}", image_R)
            elif key == ord("d"):
                # Выравнивание и сохранение изображений
                imgL, imgR = image_calibration(imgL=image_L, imgR=image_R)
                draw_epilines(imgL, imgR)
                os.makedirs(
                    f"{output_dir}/left", exist_ok=True
                )  # Создаем папку, если она не существует
                os.makedirs(
                    f"{output_dir}/right", exist_ok=True
                )  # Создаем папку, если она не существует

                cv.imwrite(f"{output_dir}/left/left{filename_L}", imgL)
                cv.imwrite(f"{output_dir}/right/right{filename_L}", imgR)
                break

        # Завершение программы
        if key == ord("q"):
            break
    cam_L.release()
    cam_R.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    """
    Основной блок программы. Инициализирует камеры и запускает обработку потока.
    """
    cam_L, cam_R = initialize_cameras()
    process_camera_stream(cam_L, cam_R)

    # Освобождение ресурсов
    cam_L.release()
    cam_R.release()
    cv.destroyAllWindows()
