from reverse_engineer_image import HexagonalDetectorGeometry
import numpy as np


def display_scandata(scandata, index, round_to=3):
    # Round x and y coordinates to 3 decimal places
    x = round(scandata['x_coord'][index], round_to)
    y = round(scandata['y_coord'][index], round_to)
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
    scandata = dd['scandata'].item()

    #print(scandata['x_coord'][6000], scandata['y_coord'][6000])
    max_index = 0
    min_index = 0
    max_data = scandata['amp0'][0]
    min_data = scandata['amp0'][0]
    for i in range(len(scandata['x_coord'])):
        if scandata['amp0'][i] > max_data:
            max_data = scandata['amp0'][i]
            max_index = i
        if scandata['amp0'][i] < min_data:
            min_data = scandata['amp0'][i]
            min_index = i
    print(max_index, min_index)
    print(max_data, scandata['x_coord'][max_index], scandata['y_coord'][max_index])
    print(min_data, scandata['x_coord'][min_index], scandata['y_coord'][min_index])
    # print(max(scandata['x_coord']), max(scandata['y_coord']))
    # print(min(scandata['x_coord']), min(scandata['y_coord']))

    # max_x: 989.6113635665853, max_y: 1012.9261904030895
    # min_x: 10.388636433414645, min_y: -12.9261904030894
