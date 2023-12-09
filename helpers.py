import numpy as np
    

def display_scandata(scandata, index, round_to=3):
    # Round x and y coordinates to 3 decimal places
    x = round(scandata["x_coord"][index], round_to)
    y = round(scandata["y_coord"][index], round_to)
    print(f"({x}, {y})")

    amp_order = [(3, 2), (4, 0, 1), (5, 6)]
    for amps in amp_order:
        amp_values = [round(scandata[f"amp{amp}"][index], round_to) for amp in amps]
        if amps != (4, 0, 1):
            print("    ", end="")
        print("    ".join(map(str, amp_values)))


dd = np.load("data/FYST_sample_1.npz", allow_pickle=True)
scandata = dd["scandata"].item()
display_scandata(scandata, 0)

print(max(scandata["x_coord"]), max(scandata["y_coord"]))
print(min(scandata["x_coord"]), min(scandata["y_coord"]))
