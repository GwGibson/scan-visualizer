from reverse_engineer_image import HexagonalDetector
import numpy as np


def display_scandata(scandata, index, round_to=3):
    # Round x and y coordinates to 3 decimal places
    x = round(scandata["x_coord"][index], round_to)
    y = round(scandata["y_coord"][index], round_to)
    print(f"({x}, {y})")

    amp_order = [(3, 2), (4, 0, 1), (5, 6)]
    for amps in amp_order:
        amp_values = [
            round(scandata[f"amp{amp}"][index], round_to) for amp in amps
        ]
        if amps != (4, 0, 1):
            print("    ", end="")
        print("    ".join(map(str, amp_values)))


if __name__ == "__main__":
    dd = np.load("data/FYST_sample_1.npz", allow_pickle=True)
    scandata = dd["scandata"].item()

    print(max(scandata["x_coord"]), max(scandata["y_coord"]))
    print(min(scandata["x_coord"]), min(scandata["y_coord"]))

    # max_x: 989.6113635665853, max_y: 1012.9261904030895
    # min_x: 10.388636433414645, min_y: -12.9261904030894
