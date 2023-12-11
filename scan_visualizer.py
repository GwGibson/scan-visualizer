import math
import numpy as np
from collections import defaultdict
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# I decided to use classes for this exercise because I was starting
# to pass around an annoying amount of variables and I wanted to keep the
# code clean and readable and add some generalization to the functionality.


# This could inherit from a more general detector class but
# do not want to get too off track for this exercise
# Also assuming a fixed number of sensors (7) here.
# It'd be fun to make this more general by allowing more and/or tighter 'rings'
class HexagonalDetectorGeometry:
    """
    Represents the geometry of a hexagonal detector array. This class is
    responsible for processing and transforming scan data based on the
    detector layout.
    """

    NUM_SENSORS = 7
    ANGLE_BETWEEN_SENSORS = math.pi / 3

    def __init__(self, sensor_distance):
        self.sensor_distance = sensor_distance
        # Intended for internal use only.
        self._scandata = None
        self._max_x, self._min_x = 0, 0
        self._max_y, self._min_y = 0, 0
        self._range_x, self._range_y = 0, 0

    def load_data(self, file_path):
        # Might be useful to allow for multiple data files to be loaded
        # and run successively through the same instance of this class.
        # Would be nice for batch processing on a server.
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

    def parse_data_for_image(self, img_width, img_height):
        if self._scandata is None:
            raise ValueError("Scan data not loaded. Please load data first.")

        # I changed the implementation here from my intial approach
        # This way is definitely more memory efficient and should be
        # faster for large datasets as well, although proper benchmarking
        # would be necessary for a conclusive answer.

        # Using defaultdict to automatically handle missing keys
        data_dict = defaultdict(lambda: [0, 0])  # Default to [sum, count]

        len_data = len(self._scandata["x_coord"])
        for index in range(len_data):
            img_coords = self._map_to_image_coords(
                index, img_width, img_height
            )
            for i, (row, col) in enumerate(img_coords):
                # If these coords exist in the dictionary, add the
                # current amplitude to the sum and increment the count
                # Otherwise, the default value of [0, 0] will be used
                data_dict[(col, row)][0] += self._scandata[f"amp{i}"][index]
                data_dict[(col, row)][1] += 1

        # Should be possible to build these arrays directly although
        # it might be a problem to search through the array to find the
        # correct index to update.
        points = np.array(
            [coords for coords, (_, count) in data_dict.items() if count > 0]
        )
        values = np.array(
            [amp / count for (amp, count) in data_dict.values() if count > 0]
        )
        return points, values

    # Mapping image coords one index at a time to avoid excessive memory usage
    # if the dataset becomes extremely large which, presumably, it will.
    def _map_to_image_coords(self, index, img_width, img_height):
        center_x, center_y = (
            self._scandata["x_coord"][index] - self._min_x,
            self._scandata["y_coord"][index] - self._min_y,
        )

        # Can probably vectorize this.
        # Would start to make sense for a large number of sensors.
        img_coords = [0 for _ in range(self.NUM_SENSORS)]
        for i in range(self.NUM_SENSORS):
            rel_x, rel_y = 0, 0
            if i != 0:  # Detector 0 is at the center
                angle = (i - 1) * self.ANGLE_BETWEEN_SENSORS
                rel_x = self.sensor_distance * math.cos(angle)
                rel_y = self.sensor_distance * math.sin(angle)
            img_x = round((center_x + rel_x) * img_width / self._range_x)
            img_y = round((center_y + rel_y) * img_height / self._range_y)
            img_coords[i] = (img_x, img_y)
        return img_coords

    # FOR TESTING PURPOSES ONLY
    # Checking that detector 0 readings are within the
    # output image range -> (1000, 1000)
    def validate_range(self):
        len_data = len(self._scandata["x_coord"])
        img_coords = [0 for _ in range(len_data)]
        for i in range(len_data):
            img_coords[i] = self._map_to_image_coords(i, 1000, 1000)
        max_pixel_x, max_pixel_y = 0, 0
        for coords in img_coords:
            if coords[0][0] > max_pixel_x:
                max_pixel_x = coords[0][0]
            if coords[0][1] > max_pixel_y:
                max_pixel_y = coords[0][1]
        return (max_pixel_x, max_pixel_y)


class ImageProcessor:
    def __init__(self, detector_geometry):
        self.detector_geometry = detector_geometry
        # Meant for internal use only or set via set_processing_params
        self._interpolation_method = "linear"
        self._sigma_value = 1
        self._griddata_method = "nearest"

    # Mainly to help me test different values
    # But could be useful for other purposes
    def set_processing_params(self, **kwargs):
        if "interpolation_method" in kwargs:
            self._interpolation_method = kwargs["interpolation_method"]
        if "sigma_value" in kwargs:
            self._sigma_value = kwargs["sigma_value"]
        if "griddata_method" in kwargs:
            self._griddata_method = kwargs["griddata_method"]

    def create_image(self, img_width, img_height):
        (points, values) = self.detector_geometry.parse_data_for_image(
            img_width, img_height
        )
        interpolated_data = self._perform_interpolation(
            points, values, img_width, img_height
        )
        image_data = self._smooth_data(interpolated_data)
        return Image.fromarray(image_data, "L")

    def _perform_interpolation(self, points, values, img_width, img_height):
        grid_x, grid_y = np.mgrid[0:img_height, 0:img_width]
        grid_z = griddata(
            points,
            values,
            (grid_x, grid_y),
            method=self._interpolation_method,
            fill_value=np.nan,
        )
        # If the initial interpolation is not nearest neighbor,
        # interpolate the missing values with nearest neighbor
        if self._interpolation_method != "nearest":
            mask = np.isnan(grid_z)
            grid_z[mask] = griddata(
                points,
                values,
                (grid_x[mask], grid_y[mask]),
                method="nearest",
            )
        return grid_z

    def _smooth_data(self, data):
        smoothed_data = gaussian_filter(data, sigma=self._sigma_value)
        smoothed_data -= np.nanmin(smoothed_data)  # Set darkest pixel to black
        if np.nanmax(smoothed_data) != 0:
            smoothed_data /= np.nanmax(smoothed_data)  # Scale to [0,1]
        smoothed_data *= 255  # Set to 8-bit grayscale
        # Catch possible floating point errors
        image_data = np.clip(smoothed_data, 0, 255).astype(np.uint8)
        return image_data


if __name__ == "__main__":
    detector = HexagonalDetectorGeometry(sensor_distance=24.552)
    try:
        detector.load_data("data/FYST_sample_1.npz")
    except FileNotFoundError as e:
        print(e)

    processor = ImageProcessor(detector)
    processor.set_processing_params(
        interpolation_method="linear",
        sigma_value=2,
        griddata_method="linear",
    )
    width, height = 1000, 1000
    processor.create_image(width, height).save("data/output.png")
