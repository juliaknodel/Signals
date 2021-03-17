import numpy as np
import cv2 as cv
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import binary_opening
from skimage.measure import label, regionprops


def get_largest_component(mask, comp_num=1):
    """
    :param mask: бинаризованное изображение
    :param comp_num: номер компоненты связности (по унименьшению площади)
    либо 1, либо 2 - третьего не дано
    :return: возвращает требуемую компоненту связности
    """
    labels = label(mask)  # разбиение маски на компоненты связности
    props = regionprops(labels)  # нахождение свойств каждой области
    areas = [prop.area for prop in props]  # площади всех компонент
    largest_comp_id = np.array(areas).argmax()

    if comp_num == 2:
        areas[largest_comp_id] = 0
        largest_comp_id = np.array(areas).argmax()

    return labels == (largest_comp_id + 1)


def get_binary_objects(img, sigma=2.5):
    """
    :param img: исходное изображение, RGB
    :param sigma: стандартное отклонение для гауссова ядра
    :return: маски для арки и блокнота
    """
    img_blur = gaussian(img, sigma=sigma, multichannel=True)

    img_blur_gray = rgb2gray(img_blur)

    # minimum оказался самым симпатичным, поэтому он везде
    thresh_min = threshold_minimum(img_blur_gray)

    # отсекаем, что не нужно
    res_min = img_blur_gray <= thresh_min

    # немножко прасив открытием (закрытие мало что изменило)
    res_min_enclosed = binary_opening(res_min, selem=np.ones((15, 15)))

    # площадь блокнота больше, чем арки - пользуемся и находим
    # в моей постановке нет мастабирования
    notepad_img = get_largest_component(res_min_enclosed, 1)
    arch_img = get_largest_component(res_min_enclosed, 2)

    # возвращаем маски
    return notepad_img, arch_img


def get_notepad_borders(mask):
    """
    :param mask: маска блокнота
    :return: границы мин. прямоугольника,
    в который может быть вписан блокнот
    """
    # индексы строк, в которых есть хотя бы 1 белый пиксель
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]

    # идексы столбцов, в которых есть хотя бы 1 белый пиксель
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    left, right = horizontal_indices[0], horizontal_indices[-1]

    return top, bottom, left, right


def get_arch_inner_borders(mask, shift=40, threshold=0.3):
    """
    :param mask: маска арки
    :param shift: сдвиг для поиска внутренней области арки
    :param threshold: максимальное число в % белых пикселей в столбце/строке
    :return: границы максимального прямоугольника,
    который может быть вписан в проход арки
    """
    # индексы строк, в которых есть хотя бы 1 белый пиксель
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]

    # идексы столбцов, в которых есть хотя бы 1 белый пиксель
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    left, right = horizontal_indices[0], horizontal_indices[-1]

    # теперь работаем внутри внешних границ арки
    min_mask = mask[top + shift:bottom - shift, left + shift:right - shift]

    # строки, в которых число белых пикселей менее 30% (порог необходим, так как не факт,
    # что до конца перекроются все дырки в арке)
    vertical_indices = np.where(np.sum(min_mask, axis=1) / min_mask.shape[1] < threshold)[0]
    inner_top, inner_bottom = vertical_indices[0] + top + shift, vertical_indices[-1] + top + shift

    # аналогично, но для столбцов
    horizontal_indices = np.where(np.sum(min_mask, axis=0) / min_mask.shape[0] < threshold)[0]
    inner_left, inner_right = horizontal_indices[0] + left + shift, horizontal_indices[-1] + left + shift

    return inner_top, inner_bottom, inner_left, inner_right


def is_fit(notepad_borders, arch_borders):
    """
    :param notepad_borders: левая/правая/верхняя/нижняя границы
    мин. прямоугольника, в который вписан блокнот
    :param arch_borders: левая/правая/верхняя/нижняя границы
    макс. прямоугольника, который может быть вписан в проход арки
    :return: Bool значение, пройет ли блокнот во внутренний проход арки,
    если можно использовать только параллельный перенос
    """
    inner_top, inner_bottom, inner_left, inner_right = arch_borders
    top_notepad, bottom_notepad, left_notepad, right_notepad = notepad_borders

    width_arch = inner_right - inner_left
    height_arch = inner_bottom - inner_top

    width_notepad = right_notepad - left_notepad
    height_notepad = bottom_notepad - top_notepad

    # по условию арка стоит ровно,
    # поэтому для параллельного переноса можем узнать ответ вот так просто:

    if width_arch > width_notepad and height_arch > height_notepad:
        return 'will fit'
    else:
        return 'no...'


def classificator(img_path):
    """
    :param img_path: путь к исходному изображению, которое
    должно удовлетворять условиям в постановке
    :return: Bool значение, пройет ли блокнот во внутренний проход арки,
    если можно использовать только параллельный перенос
    А также параметры для будущей отрисовки промежуточных результатов модели
    """
    img = cv.imread(img_path)

    # бинаризация и получение отдельно масок каждого объекта
    notepad_img, arch_img = get_binary_objects(img)

    # получение границ прямоугольников, содержащих объекты
    notepad_borders = get_notepad_borders(notepad_img)
    arch_borders = get_arch_inner_borders(arch_img)

    res = img, arch_img, notepad_img, arch_borders, notepad_borders

    # проверка влезет ли блокнот в арку и возвращаем данные для графиков
    return is_fit(notepad_borders, arch_borders), res
