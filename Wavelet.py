import numpy as np
import pywt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import namedtuple
import io_tools as io
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os

plt.ioff()

SeismogramData = namedtuple('SeismogramData', ['filename', 'trace', 'n_samples', 'scale_factor', 'sample_rate'])
Task = namedtuple('Task', ['exp_num', 'wavelet', 'start_sec', 'end_sec', 'level'])

# ======================== МАТЕМАТИКА ========================

def hilbert(signal):
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
    return np.fft.ifft(Y)

def compute_envelope(signal):
    return np.abs(hilbert(signal))

def wavelet_denoise(signal, wavelet, level, method='soft'):
    if len(signal) == 0:
        return signal
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(level, max_level)
    if level == 0:
        return signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=method))
    reconstructed = pywt.waverec(new_coeffs, wavelet)
    if len(reconstructed) > len(signal):
        reconstructed = reconstructed[:len(signal)] 
    elif len(reconstructed) < len(signal):
        reconstructed = np.pad(reconstructed, (0, len(signal) - len(reconstructed)), 'constant')
    return reconstructed

def compute_snr_astra(original_trace, filtered_trace, fs, signal_start, signal_end):
    idx_sig_start = max(0, int(round(signal_start * fs)))
    idx_sig_end = min(len(filtered_trace), int(round(signal_end * fs)))
    if idx_sig_start >= idx_sig_end:
        return float('nan')
    signal_segment = filtered_trace[idx_sig_start:idx_sig_end]
    if len(signal_segment) == 0:
        return float('nan')
    max_signal = float(np.max(np.abs(signal_segment)))
    if signal_start == 0:
        noise_start_sec = max(0, signal_end - 0.5)
        idx_noise_start = int(round(noise_start_sec * fs))
        idx_noise_end = len(original_trace)
        if idx_noise_start >= idx_noise_end:
            idx_noise_end = min(len(original_trace), int(round(0.5 * fs)))
            idx_noise_start = 0
    elif signal_start > 0.5:
        noise_end_sec = signal_start
        idx_noise_end = min(len(original_trace), int(round(noise_end_sec * fs)))
        idx_noise_start = 0
    else:
        noise_end_sec = 0.5
        idx_noise_end = min(len(original_trace), int(round(noise_end_sec * fs)))
        idx_noise_start = 0
    if idx_noise_end <= idx_noise_start:
        return float('nan')
    noise_segment = original_trace[idx_noise_start:idx_noise_end]
    if len(noise_segment) == 0:
        return float('nan')
    noise_std = float(np.std(noise_segment))
    if noise_std == 0.0:
        return float('nan')
    return float(max_signal / noise_std)

# ======================== GUI ========================

class SeismoGUI:
    def __init__(self, root):
        self.root = root
        self.seismograms = []
        self.initial_snrs = []
        self.axes = []
        self.fig = None
        self.canvas = None
        
        self.root.title("Вейвлет-фильтрация сейсмограмм")
        self.root.geometry("1300x850")
        
        style = ttk.Style()
        style.configure("TButton", padding=5, font=("Arial", 10, "bold"))
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 11, "bold"))
        
        self.control_frame = ttk.Frame(self.root, padding="15")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls()
        self.create_placeholder_plot()
        
    def setup_controls(self):
        ttk.Label(self.control_frame, text="УПРАВЛЕНИЕ", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 15))
        
        self.load_btn = ttk.Button(self.control_frame, text="📁 Загрузить файлы", command=self.load_files)
        self.load_btn.pack(fill=tk.X, pady=(0, 15))
        
        self.info_frame = ttk.LabelFrame(self.control_frame, text="Информация", padding=10)
        self.info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.files_count_label = ttk.Label(self.info_frame, text="Файлов: 0", foreground="blue")
        self.files_count_label.pack(anchor=tk.W, pady=2)
        
        self.snr_label = ttk.Label(self.info_frame, text="Изначальный S/N:\n(файлы не загружены)", 
                                   foreground="green", justify=tk.LEFT, wraplength=250)
        self.snr_label.pack(anchor=tk.W, pady=5)
        
        ttk.Label(self.control_frame, text="ПАРАМЕТРЫ АНАЛИЗА", style="Header.TLabel").pack(anchor=tk.W, pady=(15, 10))
        
        ttk.Label(self.control_frame, text="Базовый вейвлет:").pack(anchor=tk.W, pady=(5, 0))
        self.wavelet_var = tk.StringVar(value='db8')
        wavelets = ['db2', 'db4', 'db6', 'db8', 'db10', 'sym2', 'sym4', 'sym6', 'sym8', 'coif1', 'coif2', 'coif3', 'haar']
        self.wavelet_combo = ttk.Combobox(self.control_frame, textvariable=self.wavelet_var, 
                                          values=wavelets, state="readonly", width=25)
        self.wavelet_combo.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.control_frame, text="Уровень разложения:").pack(anchor=tk.W, pady=(10, 0))
        self.level_var = tk.IntVar(value=5)
        self.level_spin = ttk.Spinbox(self.control_frame, from_=1, to=15, textvariable=self.level_var, width=25)
        self.level_spin.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.control_frame, text="Начало интервала (сек):").pack(anchor=tk.W, pady=(10, 0))
        self.start_var = tk.StringVar(value="8.0")
        ttk.Entry(self.control_frame, textvariable=self.start_var, width=25).pack(fill=tk.X, pady=5)
        
        ttk.Label(self.control_frame, text="Конец интервала (сек):").pack(anchor=tk.W, pady=(10, 0))
        self.end_var = tk.StringVar(value="15.0")
        ttk.Entry(self.control_frame, textvariable=self.end_var, width=25).pack(fill=tk.X, pady=5)
        
        self.run_btn = ttk.Button(self.control_frame, text="▶ Применить фильтрацию", command=self.run_analysis)
        self.run_btn.pack(fill=tk.X, pady=25)
        
    def create_placeholder_plot(self):
        self.fig = plt.Figure(figsize=(9, 6), dpi=100)
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Нажмите "Загрузить файлы"\n для выбора сейсмотрасс', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_files(self):
        """Загрузка файлов через диалог выбора"""
        filepaths = filedialog.askopenfilenames(
            title="Выберите файлы сейсмограмм",
            filetypes=[("Сейсмические файлы", "*.01x *.01y *.01z *.dat *.bin"), ("Все файлы", "*.*")]
        )
        if not filepaths:
            return
        
        self.seismograms = []
        for fp in filepaths:
            data = io.load_data(fp, triger=0)
            if data[1] == 0:
                self.seismograms.append(SeismogramData(fp, data[0], *data[2:]))
                print(f"(OK) Загружен: {os.path.basename(fp)}")
            else:
                print(f"(ERROR) Не загружен: {os.path.basename(fp)}")
        
        if not self.seismograms:
            messagebox.showerror("Ошибка", "Не удалось загрузить ни одного файла!")
            return
        
        self.compute_initial_snrs()
        self.update_info_display()
        self.create_plots()
        self.draw_original_traces()
        
    def compute_initial_snrs(self):
        """Вычисляет изначальный S/N для всех загруженных трасс"""
        self.initial_snrs = []
        try:
            start_sec = float(self.start_var.get().replace(',', '.'))
            end_sec = float(self.end_var.get().replace(',', '.'))
        except ValueError:
            start_sec, end_sec = 8.0, 15.0
            
        for seism in self.seismograms:
            snr = compute_snr_astra(seism.trace, seism.trace, seism.sample_rate, start_sec, end_sec)
            self.initial_snrs.append(snr)
    
    def update_info_display(self):
        """Обновляет информационную панель"""
        self.files_count_label.config(text=f"Файлов: {len(self.seismograms)}")
        snr_lines = []
        for i, (seism, snr) in enumerate(zip(self.seismograms, self.initial_snrs)):
            base = os.path.basename(seism.filename)
            comp = base.split('.')[-1][-1].upper() if '.' in base else f'#{i+1}'
            try:
                if not np.isnan(snr):
                    snr_lines.append(f"  {comp}: {snr:.2f}")
                else:
                    snr_lines.append(f"  {comp}: N/A")
            except:
                snr_lines.append(f"  {comp}: N/A")
        snr_text = "Изначальный S/N:\n" + "\n".join(snr_lines) if snr_lines else "Изначальный S/N:\n(не вычислен)"
        self.snr_label.config(text=snr_text)
        
    def create_plots(self):
        """Создаёт область для графиков"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        n_files = len(self.seismograms)
        self.fig, self.axes = plt.subplots(n_files, 1, figsize=(9, 2.5 * n_files + 1), sharex=True, dpi=100)
        if n_files == 1:
            self.axes = [self.axes]
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def draw_original_traces(self):
        """Отрисовка исходных трасс с начальным S/N"""
        for i, ax in enumerate(self.axes):
            ax.clear()
            seism = self.seismograms[i]
            trace = seism.trace
            fs = seism.sample_rate
            t = np.arange(len(trace)) / fs
            
            base = os.path.basename(seism.filename)
            comp = base.split('.')[-1][-1].upper() if '.' in base else f'#{i+1}'
            
            ax.plot(t, trace, 'b-', linewidth=0.8, label='Исходный')
            
            snr = self.initial_snrs[i] if i < len(self.initial_snrs) else float('nan')
            try:
                snr_str = f'{float(snr):.2f}' if not np.isnan(float(snr)) else 'nan'
            except:
                snr_str = 'nan'
            
            ax.set_title(f'Компонента {comp} | Изначальный SNR = {snr_str}')
            ax.set_ylabel('Амплитуда')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        self.axes[-1].set_xlabel('Время (с)')
        self.fig.suptitle('Исходные сейсмограммы', fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()
        
    def run_analysis(self):
        """Запускает анализ с фильтрацией"""
        if not self.seismograms:
            messagebox.showwarning("Warning", "Сначала загрузите файлы!")
            return
            
        try:
            wavelet = self.wavelet_var.get()
            level = int(self.level_var.get())
            start_sec = float(self.start_var.get().replace(',', '.'))
            end_sec = float(self.end_var.get().replace(',', '.'))
            if start_sec >= end_sec:
                raise ValueError("Начало должно быть меньше конца")
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Проверьте корректность введённых значений.\n{e}")
            return
        
        task = Task(1, wavelet, start_sec, end_sec, level)
        self.root.config(cursor="watch")
        self.root.update()
        
        results = self.process_task(task)
        self.draw_filtered_results(results)
        self.root.config(cursor="")
        
    def process_task(self, task):
        """Выполняет фильтрацию для всех трасс"""
        filtered_traces = []
        envelopes = []
        snr_values = []
        
        for seismogram in self.seismograms:
            trace = seismogram.trace
            fs = seismogram.sample_rate
            idx_start = max(0, int(round(task.start_sec * fs)))
            idx_end = min(len(trace), int(round(task.end_sec * fs)))
            
            if idx_start >= idx_end:
                filtered_traces.append(trace.copy())
                envelopes.append(compute_envelope(trace))
                snr_values.append(float('nan'))
                continue
                
            segment = trace[idx_start:idx_end].copy()
            filtered_segment = wavelet_denoise(segment, task.wavelet, task.level)
            trace_filtered = trace.copy()
            trace_filtered[idx_start:idx_end] = filtered_segment
            snr = compute_snr_astra(trace, trace_filtered, fs, task.start_sec, task.end_sec)
            envelope = compute_envelope(trace_filtered)
            filtered_traces.append(trace_filtered)
            envelopes.append(envelope)
            snr_values.append(snr)
            
        return {
            'task': task,
            'filtered_traces': filtered_traces,
            'envelopes': envelopes,
            'snr_values': snr_values
        }
        
    def draw_filtered_results(self, res):
        """Отрисовка результатов фильтрации"""
        task = res['task']
        for i, ax in enumerate(self.axes):
            ax.clear()
            seism = self.seismograms[i]
            trace = seism.trace
            fs = seism.sample_rate
            t = np.arange(len(trace)) / fs
            
            base = os.path.basename(seism.filename)
            comp = base.split('.')[-1][-1].upper() if '.' in base else f'#{i+1}'
            
            filtered = res['filtered_traces'][i]
            envelope = res['envelopes'][i]
            snr = res['snr_values'][i]
            
            xmin = max(0, task.start_sec - 2)
            xmax = min(t[-1], task.end_sec + 2)
            mask = (t >= xmin) & (t <= xmax)
            
            ax.plot(t[mask], trace[mask], 'g-', label='Исходный', linewidth=0.8, alpha=0.6)
            ax.plot(t[mask], filtered[mask], 'b-', label='Фильтрованный', linewidth=1.0)
            ax.plot(t[mask], envelope[mask], 'r-', label='Огибающая', linewidth=0.8)
            ax.axvspan(task.start_sec, task.end_sec, color='yellow', alpha=0.3, label='Окно анализа')
            
            try:
                snr_str = f'{float(snr):.2f}' if not np.isnan(float(snr)) else 'nan'
            except:
                snr_str = 'nan'
            ax.set_title(f'Компонента {comp} | SNR = {snr_str}')
            ax.set_ylabel('Амплитуда')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        self.axes[-1].set_xlabel('Время (с)')
        self.fig.suptitle(f'Вейвлет: {task.wavelet} | Уровень: {task.level} | Интервал: [{task.start_sec}, {task.end_sec}] с', fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = SeismoGUI(root)  # Без автозагрузки — файлы выбираются вручную
    root.mainloop()

if __name__ == "__main__":
    main()