import numpy as np
import struct
import os

def load_data(filename, start_time=0, duration=-1, triger=0):
    """
    Чтение сейсмических данных из файла в PC-формате.
    """
    err = 0
    trace = np.array([])
    n_samples = 0
    scale_factor = 0.0
    sample_rate = 0

    try:
        with open(filename, 'rb') as file:
            print(f"(УВЕДОМЛЕНИЕ) Чтение файла {filename}.")
            
            # 1. Читаем sample_type (2 байта) по смещению 38:
            file.seek(38)
            sample_type = struct.unpack('<h', file.read(2))[0]
            
            # 2. Читаем scale_factor (8 байт, double) по смещению 14:
            file.seek(14)
            scale_factor = struct.unpack('<d', file.read(8))[0]
            
            # 3. Читаем sample_rate (2 байта, unsigned short) по смещению 32:
            file.seek(32)
            sample_rate = struct.unpack('<H', file.read(2))[0]
            
            bytes_per_sample = sample_type & 15
            
            if bytes_per_sample not in (2, 4, 8):
                raise ValueError(f"(ОШИБКА, ФАЙЛ) Неверный размер отсчёта: {bytes_per_sample}")
            
            offset = round(start_time * sample_rate) * bytes_per_sample + 42
            file.seek(offset)
            
            if duration > 0:
                n_samples_to_read = round(duration * sample_rate)
            else:
                current = file.tell()
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                remaining = file_size - current
                n_samples_to_read = remaining // bytes_per_sample
                file.seek(current)
            
            dtype_map = {
                2: np.int16,
                258: np.uint16,
                4: np.int32,
                4100: np.float32,
                4104: np.float64
            }
            if sample_type not in dtype_map:
                raise ValueError(f"(ОШИБКА, ФАЙЛ) Неизвестный тип данных: {sample_type}")
            dtype = dtype_map[sample_type]
            
            data_bytes = file.read(n_samples_to_read * bytes_per_sample)
            
            if len(data_bytes) == 0:
                err = 2
            else:
                trace = np.frombuffer(data_bytes, dtype=dtype).astype(np.float64)
                n_samples = len(trace)
                
                if duration > 0 and n_samples < n_samples_to_read:
                    err = 2
                
                if scale_factor != 0:
                    if triger == 1:
                        trace = (trace / 32767.0) * scale_factor
                    elif triger == 2:
                        trace = trace * scale_factor
                        
    except FileNotFoundError:
        err = 1
    except Exception:
        err = 2

    return trace, err, n_samples, scale_factor, sample_rate