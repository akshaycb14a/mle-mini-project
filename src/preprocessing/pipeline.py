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

        if self.window.ready() and self.sample_count % 10 == 0:

            features = FeatureExtractor.extract(
                list(self.window.temperature),
                list(self.window.vibration)
            )

            self.dataset.append(
                features,
                data["status"]
            )

            print("Saved feature vector:")
            print(features)

            return features

        return None