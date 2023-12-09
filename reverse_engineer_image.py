import numpy as np


# This could inherit from a more general detector class but
# do not want to get too off track for this exercise
# Also assuming a fixed number of sensors (7) here.
# We could make this more general by allowing more 'rings'
class HexagonalDetector:
    def __init__(self, detector_distance):
        self.detector_distance = detector_distance
        self.scandata = None
        self.img_width = 0
        self.img_height = 0

    def load_data(self, file_path):
        try:
            dd = np.load(file_path, allow_pickle=True)
            self.scandata = dd["scandata"].item()
        except FileNotFoundError as ex:
            print(f"An error occurred while loading the data: {ex}")
        self.img_width = max(self.scandata["x_coord"]) - min(
            self.scandata["x_coord"]
        )
        self.img_height = max(self.scandata["y_coord"]) - min(
            self.scandata["y_coord"]
        )


if __name__ == "__main__":
    detector = HexagonalDetector(detector_distance=24.552)
    try:
        detector.load_data("data/FYST_sample_1.npz")
    except FileNotFoundError as e:
        print(e)
