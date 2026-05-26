import numpy as np
import pywt
import matplotlib.pyplot as plt
from collections import namedtuple
import io_tools as io




# Структура данных сейсмограммы:
SeismogramData = namedtuple('SeismogramData', ['filename', 'trace', 'n_samples', 'scale_factor', 'sample_rate'])

# Структура задания:
Task = namedtuple('Task', ['exp_num', 'wavelet', 'start_sec', 'end_sec', 'level'])

# ========== ПАРАМЕТРЫ ==========

filenames = [
    '11-05/12301105.01x',
    '11-05/12301105.01y',
    '11-05/12301105.01z']

tasks = [
    Task(1, 'db8', 0, 10, 8),]


# ======================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========================

def hilbert(signal):
    """
    Вычисляет аналитический сигнал через преобразование Гильберта.
    Используется для получения огибающей.

    Параметры:
        signal : np.ndarray - входной вещественный сигнал

    Возвращает:
        analytic : np.ndarray - комплексный аналитический сигнал
    """
    N = len(signal)
    X = np.fft.fft(signal)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    Y = X * h
    analytic = np.fft.ifft(Y)
    return analytic


def compute_envelope(signal):
    """
    Вычисляет огибающую вещественного сигнала как модуль аналитического сигнала.

    Параметры:
        signal : np.ndarray - входной сигнал

    Возвращает:
        envelope : np.ndarray - огибающая (действительная)
    """
    analytic = hilbert(signal)
    return np.abs(analytic)


def wavelet_denoise(signal, wavelet, level, method='soft'):
    """
    Выполняет вейвлет-фильтрацию сигнала с пороговой обработкой коэффициентов.

    Параметры:
        signal  : np.ndarray - одномерный сигнал
        wavelet : str        - имя вейвлета (например, 'db8')
        level   : int        - уровень разложения
        method  : str        - тип порога: 'soft' (мягкий) или 'hard' (жёсткий)

    Возвращает:
        filtered : np.ndarray - отфильтрованный сигнал той же длины, что и входной
    """
    # Если сигнал пустой — возвращаем его без изменений:
    if len(signal) == 0:
        return signal

    # Максимально возможный уровень разложения для данного сигнала и вейвлета:
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    # Ограничиваем реальный уровень разложения максимальным:
    level = min(level, max_level)

    # Если уровень стал нулевым — фильтрация невозможна:
    if level == 0:
        print("(ПРЕДУПРЕЖДЕНИЕ) Уровень разложения слишком велик, фильтрация не выполнена")
        return signal

    # Прямое вейвлет-преобразование: список массивов коэффициентов:
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Оценка уровня шума по медиане абсолютных отклонений (MAD) детализирующих коэффициентов самого высокого уровня:
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # Универсальный порог (Universal Threshold):
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Начинаем формировать новый список коэффициентов: первый элемент — аппроксимирующие коэффициенты (не трогаем):
    new_coeffs = [coeffs[0]]
    # Для всех уровней детализирующих коэффициентов применяем пороговую обработку:
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=method))

    # Обратное вейвлет-преобразование для восстановления сигнала:
    reconstructed = pywt.waverec(new_coeffs, wavelet)

    # Длина восстановленного сигнала может отличаться на один отсчёт из-за особенностей свёртки:
    if len(reconstructed) > len(signal): # Обрезаем лишнее:
        reconstructed = reconstructed[:len(signal)] 
    elif len(reconstructed) < len(signal): # Дополняем нулями, если не хватает:
        reconstructed = np.pad(reconstructed, (0, len(signal) - len(reconstructed)), 'constant')
    return reconstructed


def compute_snr_astra(original_trace, filtered_trace, fs, signal_start, signal_end):
    """
    Вычисляет отношение сигнал/шум по методике системы «Астра».

    Параметры:
        original_trace  : np.ndarray - исходная сейсмограмма (для оценки шума)
        filtered_trace  : np.ndarray - отфильтрованная сейсмограмма (для сигнала)
        fs              : int        - частота дискретизации (Гц)
        signal_start    : float      - начало сигнального интервала (сек)
        signal_end      : float      - конец сигнального интервала (сек)

    Возвращает:
        snr : float - отношение максимальной амплитуды сигнала к СКО шума
    """ 
    # Индексы начала и конца обрабатываемого временного окна:
    idx_sig_start = max(0, int(round(signal_start * fs)))  # Переводим секунды в отсчёты, округляем, не допускаем отрицательных значений
    idx_sig_end   = min(len(filtered_trace), int(round(signal_end * fs)))  # Верхняя граница: не выходить за длину массива

    # Если окно пустое или отрицательное — прекращаем работу:
    if idx_sig_start >= idx_sig_end: 
        return np.nan

    # Вырезаем сегмент из оригинального сигнала и выбираем на нём максимальное значение:
    signal_segment = filtered_trace[idx_sig_start:idx_sig_end]  # Извлекаем фрагмент фильтрованной сейсмограммы в сигнальном окне;
    max_signal = np.max(np.abs(signal_segment))  # Определяем максимальную амплитуду (модуль) среди этого фрагмента;

    if signal_start > 0.5:  # Если начало сигнала позже 0.5 секунды, шум можно оценить на участке до начала сигнала:
        noise_end_sec = signal_start  # Конец шумового интервала по времени – непосредственно перед сигналом;
    else:  # Если сигнал начинается раньше 0.5 с, шум оцениваем на первых 0.5 секундах записи:
        noise_end_sec = 0.5  # Фиксируем конец шумового интервала в 0.5 секунды;

    idx_noise_end = min(len(original_trace), int(round(noise_end_sec * fs)))  # Переводим границу шума в отсчёты, не выходя за длину исходного массива;
    if idx_noise_end <= 0:  # Проверяем, что получен хотя бы один отсчёт шума:
        return np.nan  # Если шумовой интервал пуст – возвращаем NaN;

    noise_segment = original_trace[0:idx_noise_end]  # Вырезаем начальный фрагмент исходной сейсмограммы для оценки шума;
    noise_std = np.std(noise_segment)  # Вычисляем среднеквадратическое отклонение (СКО) шума;
    if noise_std == 0:  # Если шум отсутствует (нулевое СКО), считаем SNR бесконечным:
        return np.inf  # Возвращаем положительную бесконечность;

    return max_signal / noise_std  # Возвращаем отношение максимального сигнала к СКО шума;

# ======================== ВЫЧИСЛЕНИЕ ВСЕХ ЭКСПЕРИМЕНТОВ ========================

def compute_all_experiments(seismograms, tasks):
    """
    Выполняет вейвлет-фильтрацию для всех экспериментов и сохраняет результаты.

    Параметры:
        seismograms : list - загруженные сейсмограммы
        tasks : list of namedtuple - список заданий

    Возвращает:
        results : list of dict - для каждого эксперимента хранит:
            - 'task': исходная задача
            - 'filtered_traces' : list of np.ndarray
            - 'envelopes' : list of np.ndarray
            - 'snr_values' : list of float
    """
    results = []                           # Список для результатов всех экспериментов;

    for task in tasks:                     # Перебираем все задания (эксперименты):
        print(f"Вычисление эксперимента {task.exp_num}...")

        filtered_traces = []               # Список отфильтрованных трасс для текущего эксперимента;
        envelopes = []                     # Список огибающих для каждой трассы;
        snr_values = []                    # Список значений SNR;

        for seismogram in seismograms:          # Обрабатываем каждую сейсмограмму:
            trace = seismogram.trace            # Исходный сигнал (np.ndarray);
            fs = seismogram.sample_rate         # Частота дискретизации, Гц;

            # Индексы начала и конца обрабатываемого временного окна:
            idx_start = max(0, int(round(task.start_sec * fs)))
            idx_end = min(len(trace), int(round(task.end_sec * fs)))

            # Если окно пустое или отрицательное — сохраняем исходную трассу без фильтрации:
            if idx_start >= idx_end:
                filtered_traces.append(trace.copy())
                envelopes.append(compute_envelope(trace))
                snr_values.append(np.nan)
                continue                     # Переходим к следующей сейсмограмме;

            # Вырезаем сегмент и применяем вейвлет-фильтрацию:
            segment = trace[idx_start:idx_end].copy()
            filtered_segment = wavelet_denoise(segment, task.wavelet, task.level)

            # Создаём копию полной трассы и заменяем в ней обработанный сегмент:
            trace_filtered = trace.copy()
            trace_filtered[idx_start:idx_end] = filtered_segment

            # Вычисляем SNR (отношение сигнал/шум) для изменённого участка:
            snr = compute_snr_astra(trace, trace_filtered, fs, task.start_sec, task.end_sec)

            # Вычисляем огибающую отфильтрованной трассы:
            envelope = compute_envelope(trace_filtered)

            # Сохраняем результаты для текущей сейсмограммы:
            filtered_traces.append(trace_filtered)
            envelopes.append(envelope)
            snr_values.append(snr)

        # После обработки всех сейсмограмм сохраняем данные эксперимента в results
        results.append({
            'task': task,
            'filtered_traces': filtered_traces,
            'envelopes': envelopes,
            'snr_values': snr_values
        })

    return results # Возвращаем список со словарями результатов для всех экспериментов;

# ======================== ГЛАВНАЯ ФУНКЦИЯ ========================

def run(filenames, tasks, scale_mode=0):
    """
    Параметры:
        filenames : list of str - пути к файлам сейсмограмм
        tasks     : list of namedtuple Task - список заданий
        scale_mode : int - режим масштабирования (0 - без изменений, 1 - коррелограмма, 2 - умножение)
    """

    # Пытаемся считать сейсмограммы из файлов:
    seismograms = [
        SeismogramData(filename, data[0], *data[2:]) # (filename, trace, n_samples, scale_factor, sample_rate)
        for filename in filenames if (data := io.load_data(filename, triger=scale_mode))[1] == 0]

    # Проверка на наличие сейсмограмм:
    if not seismograms:
        print("(ОШИБКА) Не удалось прочитать файлы!")
        return None  # Завершаем функцию;
    
    print(f"(УВЕДОМЛЕНИЕ) Загружено {len(seismograms)} сейсмограмм.")

    print("(УВЕДОМЛЕНИЕ) Вычисление всех экспериментов...")
    results = compute_all_experiments(seismograms, tasks)

    print("(УВЕДОМЛЕНИЕ) Запуск интерактивного просмотра.")
    viewer = io.InteractiveViewer(seismograms, results)
    plt.show()


# ======================== ТОЧКА ВХОДА ========================

run(filenames, tasks, scale_mode=0)