import math
import numpy as np


# This could inherit from a more general detector class but
# do not want to get too off track for this exercise
# Also assuming a fixed number of sensors (7) here.
# We could make this more general by allowing more 'rings'
class HexagonalDetector:
    NUM_SENSORS = 7

    def __init__(self, detector_distance):
        self.detector_distance = detector_distance
        # Intended for internal use only.
        self._scandata = None
        self._max_x, self._min_x = 0, 0
        self._max_y, self._min_y = 0, 0
        self._range_x, self._range_y = 0, 0

    def load_data(self, file_path):
        try:
            dd = np.load(file_path, allow_pickle=True)
            self._scandata = dd["scandata"].item()
            self._min_x, self._min_y = min(self._scandata["x_coord"]), min(
                self._scandata["y_coord"]
            )
            self._max_x, self._max_y = max(self._scandata["x_coord"]), max(
                self._scandata["y_coord"]
            )
            self._range_x, self._range_y = (
                self._max_x - self._min_x,
                self._max_y - self._min_y,
            )
        except FileNotFoundError as ex:
            print(f"An error occurred while loading the data: {ex}")

    def create_image(self, img_width, img_height):
        if self._scandata is None:
            raise ValueError("Scan data not loaded. Please load data first.")
        return self.__map_to_image_coords(0, img_width, img_height)

    # Mapping image coords one index at a time to avoid excessive memory usage
    # if the dataset becomes extremely large which, presumably, it will.
    def __map_to_image_coords(self, index, img_width, img_height):
        center_x, center_y = (
            self._scandata["x_coord"][index] - self._min_x,
            self._scandata["y_coord"][index] - self._min_y,
        )

        img_coords = [0 for _ in range(self.NUM_SENSORS)]
        for i in range(self.NUM_SENSORS):
            rel_x, rel_y = 0, 0
            if i != 0:
                angle = (i - 1) * (math.pi / 3)
                rel_x = self.detector_distance * math.cos(angle)
                rel_y = self.detector_distance * math.sin(angle)
            img_x = round((center_x + rel_x) * img_width / self._range_x)
            img_y = round((center_y + rel_y) * img_height / self._range_y)
            img_coords[i] = (img_x, img_y)
        return img_coords

    # FOR TESTING PURPOSES ONLY
    # Checking that detector 0 readings are within the
    # output image range -> (1000, 1000)
    def validate_range(self):
        img_coords = [0 for _ in range(len(self._scandata["x_coord"]))]
        for i in range(len(self._scandata["x_coord"])):
            img_coords[i] = self.__map_to_image_coords(i, 1000, 1000)
        max_pixel_x, max_pixel_y = 0, 0
        for coords in img_coords:
            if coords[0][0] > max_pixel_x:
                max_pixel_x = coords[0][0]
            if coords[0][1] > max_pixel_y:
                max_pixel_y = coords[0][1]
        return (max_pixel_x, max_pixel_y)


if __name__ == "__main__":
    detector = HexagonalDetector(detector_distance=24.552)
    try:
        detector.load_data("data/FYST_sample_1.npz")
    except FileNotFoundError as e:
        print(e)

    # print(detector.create_image(1000, 1000))
    print(detector.validate_range())
