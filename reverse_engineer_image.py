import math
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PIL import Image

# I decided to use classes for this exercise because I was starting
# to pass around an annoying amount of variables and I wanted to keep the
# code clean and readable and add some generalization to the functionality.


# This could inherit from a more general detector class but
# do not want to get too off track for this exercise
# Also assuming a fixed number of sensors (7) here.
# It'd be fun to make this more general by allowing more or tighter 'rings'
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
        # Sparse matrices might make sense here
        image_data = np.zeros((img_height, img_width))
        count_data = np.zeros((img_height, img_width))

        len_data = len(self._scandata["x_coord"])
        for index in range(len_data):
            img_coords = self._map_to_image_coords(
                index, img_width, img_height
            )
            for i, coord in enumerate(img_coords):
                row, col = coord
                if 0 <= row < img_width and 0 <= col < img_height:
                    # [row, col] seems to produce incorrect output
                    # Apparently images are typically represented with the
                    # first dimension as rows and the second dimension as
                    # columns, but our coordinates are based on an (x, y)
                    # Cartesian system. Hence, we need to reverse the order
                    # when indexing into the image data array.
                    image_data[col, row] += self._scandata[f"amp{i}"][index]
                    count_data[col, row] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            averaged_data = np.true_divide(image_data, count_data)
            averaged_data[~np.isfinite(averaged_data)] = 0
        # Need to count_data for proper interpolation since
        # a '0' measurement is not the same as a missing measurement
        return (averaged_data, count_data)

    # Mapping image coords one index at a time to avoid excessive memory usage
    # if the dataset becomes extremely large which, presumably, it will.
    def _map_to_image_coords(self, index, img_width, img_height):
        if self._scandata is None:
            raise ValueError("Scan data not loaded. Please load data first.")

        center_x, center_y = (
            self._scandata["x_coord"][index] - self._min_x,
            self._scandata["y_coord"][index] - self._min_y,
        )

        # Can probably vectorize this.
        # Would start to make sense for a large number of sensors.
        img_coords = [0 for _ in range(self.NUM_SENSORS)]
        for i in range(self.NUM_SENSORS):
            rel_x, rel_y = 0, 0
            if i != 0:
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
        (data, count_data) = self.detector_geometry.parse_data_for_image(
            img_width, img_height
        )
        # Prepare data for interpolation
        # Get rid of zero COUNT values (areas the detectors did not cover)
        points = np.argwhere(count_data > 0)  # Coords of measured areas
        values = data[count_data > 0]  # Corresponding measured values

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
        # Seems there are still NaN values after interpolation
        # 'nearest' neighbor will resolve all NaN values but
        # user can specify linear/cubic interpolation for griddata
        # which may not resolve all NaN values
        mask = np.isnan(grid_z)
        grid_z[mask] = griddata(
            points,
            values,
            (grid_x[mask], grid_y[mask]),
            method=self._griddata_method,
        )
        return grid_z

    def _smooth_data(self, data):
        smoothed_data = gaussian_filter(data, sigma=self._sigma_value)
        # Replace 'nan' values with the minimum value after smoothing
        # Necessary if 'griddata_method' is not 'nearest'
        min_value = np.nanmin(smoothed_data[np.isfinite(smoothed_data)])
        smoothed_data = np.nan_to_num(smoothed_data, nan=min_value)

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
    processor.create_image(1000, 1000).save("output.png")
