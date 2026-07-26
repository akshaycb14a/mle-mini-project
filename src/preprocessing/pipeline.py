from collections import Counter

from src.preprocessing.moving_average import MovingAverageFilter
from src.preprocessing.dataset_builder import SlidingWindow, DatasetBuilder
from src.preprocessing.feature_extractor import FeatureExtractor


class PreprocessingPipeline:

    def __init__(self):

        self.temp_filter = MovingAverageFilter(window_size=5)
        self.vib_filter = MovingAverageFilter(window_size=5)

        self.window = SlidingWindow(size=30)

        self.dataset = DatasetBuilder(
            "data/processed/training_dataset.csv"
        )

        self.sample_count = 0

    def process(self, data):
        """
        Process one incoming MQTT sensor reading.
        """

        filtered_temp = self.temp_filter.update(
            data["temperature"]
        )

        filtered_vib = self.vib_filter.update(
            data["vibration"]
        )

        self.window.add(
            filtered_temp,
            filtered_vib,
            data["door_open"],
            data["status"]
        )

        self.sample_count += 1

        print(
            f"sample_count={self.sample_count}, "
            f"window_size={len(self.window.temperature)}, "
            f"ready={self.window.ready()}"
        )

        if self.window.ready() and self.sample_count % 10 == 0:

            features = FeatureExtractor.extract(
                list(self.window.temperature),
                list(self.window.vibration)
            )

            window_label = Counter(self.window.status).most_common(1)[0][0]

            self.dataset.append(
                features,
                window_label
            )

            print("Saved feature vector:")
            print(features)
            print(f"Window Label: {window_label}")

            return features

        return None