import numpy as np


class FeatureExtractor:

    @staticmethod
    def extract(temp_window, vib_window):

        temp = np.array(temp_window)
        vib = np.array(vib_window)

        temp_mean = np.mean(temp)

        temp_std = np.std(temp)

        temp_rate = (temp[-1] - temp[0]) * 2
        # 30-second window → convert to °C/min

        vibration_rms = np.sqrt(np.mean(vib ** 2))

        vibration_peak = np.max(vib)

        vibration_kurtosis = (
            np.mean((vib - np.mean(vib)) ** 4)
            / (np.std(vib) ** 4 + 1e-8)
        )

        return {
            "temp_mean": temp_mean,
            "temp_std": temp_std,
            "temp_rate": temp_rate,
            "vibration_rms": vibration_rms,
            "vibration_peak": vibration_peak,
            "vibration_kurtosis": vibration_kurtosis,
        }

if __name__ == "__main__":

    temps = np.random.normal(4, 0.3, 30)

    vibs = np.random.normal(0.8, 0.1, 30)

    features = FeatureExtractor.extract(
        temps,
        vibs
    )

    print(features)