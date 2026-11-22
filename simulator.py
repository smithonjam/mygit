import numpy as np
import scipy.fftpack as fft
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json
from datetime import datetime


class SSVEP_Simulator:
    def __init__(self, sample_rate: int = 1000, duration: float = 5.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.total_samples = int(self.sample_rate * self.duration)
        self.time_axis = np.linspace(0, self.duration, self.total_samples, endpoint=False)
        self.noise_amplitude = 0.8
        self.alpha_amplitude = 0.3
        self.baseline_drift = 0.2
        self.storage_path = "./ssvep_records"
        self.log_file = os.path.join(self.storage_path, "experiment_log.json")
        self._init_folder()

    def _init_folder(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        if not os.path.exists(self.log_file):
            init_log = {"records": []}
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(init_log, f, ensure_ascii=False, indent=2)

    def generate_eye_blink_artifact(self, n_samples: int, blink_count: int = 3):
        artifact = np.zeros(n_samples)
        blink_positions = np.random.choice(np.arange(100, n_samples - 100), size=blink_count, replace=False)
        for pos in blink_positions:
            blink_window = np.arange(pos - 30, pos + 30)
            blink_window = blink_window[(blink_window >= 0) & (blink_window < n_samples)]
            gauss = np.exp(-((blink_window - pos) / 12) ** 2) * 12
            artifact[blink_window] += gauss
        return artifact

    def generate_ssvep_signal(self, target_freq: float, snr: float = None):
        clean_ssvep = np.sin(2 * np.pi * target_freq * self.time_axis) * 1.2
        alpha_wave = np.sin(2 * np.pi * 10 * self.time_axis) * self.alpha_amplitude
        gauss_noise = np.random.normal(loc=0, scale=self.noise_amplitude, size=self.total_samples)
        slow_drift = self.baseline_drift * np.sin(2 * np.pi * 0.08 * self.time_axis)
        blink_artifact = self.generate_eye_blink_artifact(self.total_samples, blink_count=4)
        raw_signal = clean_ssvep + alpha_wave + gauss_noise + slow_drift + blink_artifact
        return raw_signal, clean_ssvep

    def compute_fft_spectrum(self, signal_data: np.ndarray):
        fft_vals = fft.fft(signal_data)
        fft_freq = fft.fftfreq(self.total_samples, d=1 / self.sample_rate)
        amp = 2.0 / self.total_samples * np.abs(fft_vals)
        positive_mask = fft_freq >= 0
        return fft_freq[positive_mask], amp[positive_mask]

    def filter_bandpass(self, sig: np.ndarray, low: float = 1.0, high: float = 50.0):
        b, a = signal.butter(4, [low, high], btype="bandpass", fs=self.sample_rate)
        filtered = signal.filtfilt(b, a, sig)
        return filtered

    def save_signal_csv(self, sig: np.ndarray, freq: float, note: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"ssvep_{freq}hz_{timestamp}.csv"
        csv_path = os.path.join(self.storage_path, csv_name)
        df = pd.DataFrame({
            "time_s": self.time_axis,
            "eeg_signal": sig
        })
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log_entry = {
            "time": timestamp,
            "target_frequency": freq,
            "file": csv_name,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "note": note
        }
        with open(self.log_file, "r+", encoding="utf-8") as f:
            log_data = json.load(f)
            log_data["records"].append(log_entry)
            f.seek(0)
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return csv_path

    def plot_signal_and_spectrum(self, raw_sig, filtered_sig, freq_axis, amp_axis, target_freq):
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        axes[0].plot(self.time_axis, raw_sig, color="#2c3e50", linewidth=0.8)
        axes[0].set_title(f"原始EEG信号 | 目标SSVEP频率: {target_freq} Hz", fontsize=12)
        axes[0].set_xlabel("时间 (s)")
        axes[0].set_ylabel("幅值 (μV)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.time_axis, filtered_sig, color="#e74c3c", linewidth=0.9)
        axes[1].set_title("带通滤波后信号 (1~50Hz)", fontsize=12)
        axes[1].set_xlabel("时间 (s)")
        axes[1].set_ylabel("幅值 (μV)")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(freq_axis, amp_axis, color="#27ae60", linewidth=1.0)
        axes[2].axvline(x=target_freq, color="orange", linestyle="--", label=f"目标 {target_freq} Hz")
        axes[2].set_xlim(0, 45)
        axes[2].set_title("频域FFT频谱", fontsize=12)
        axes[2].set_xlabel("频率 (Hz)")
        axes[2].set_ylabel("振幅")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        img_path = os.path.join(self.storage_path, f"plot_{target_freq}hz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(img_path, dpi=200, bbox_inches="tight")
        plt.show()
        return img_path

    def run_single_trial(self, target_frequency: float):
        print(f"\n===== 开始单次试次 | 刺激频率 {target_frequency} Hz =====")
        raw, clean = self.generate_ssvep_signal(target_frequency)
        filtered = self.filter_bandpass(raw, 1, 50)
        f_freq, f_amp = self.compute_fft_spectrum(filtered)
        csv_p = self.save_signal_csv(raw, target_frequency, note="single trial test")
        fig_p = self.plot_signal_and_spectrum(raw, filtered, f_freq, f_amp, target_frequency)
        print(f"数据保存至: {csv_p}")
        print(f"图谱保存至: {fig_p}")
        return raw, filtered, f_freq, f_amp

    def run_batch_experiment(self, freq_list: list):
        print("\n======== 批量SSVEP实验启动 ========")
        for f in freq_list:
            self.run_single_trial(f)
            time.sleep(1.2)
        print("===== 全部试次结束 =====")

    def load_log_and_summary(self):
        with open(self.log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
        df_log = pd.DataFrame(log["records"])
        print("\n======== 实验日志汇总 ========")
        print(df_log)
        return df_log


def main():
    simulator = SSVEP_Simulator(sample_rate=1000, duration=6.0)
    stimulus_frequencies = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    simulator.run_batch_experiment(stimulus_frequencies)
    simulator.load_log_and_summary()
    print("\n程序运行完毕！所有EEG仿真数据、图片、日志都在 ./ssvep_records 文件夹")


if __name__ == "__main__":
    main()
