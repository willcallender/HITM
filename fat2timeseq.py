import numpy as np
from matplotlib import pyplot as plt

filename = 'firespreaddata/firstarrival.txt'

data = np.loadtxt(filename)

def get_spread_mask(timestamps):
    spread_mask = np.zeros((len(timestamps), *data.shape), dtype=bool)
    for i, timestamp in enumerate(timestamps):
        spread_mask[i, :] = data <= timestamp
    return spread_mask

timestamps = np.arange(101) # np.linspace(0, 100, 201)
spread_masks = get_spread_mask(timestamps)

# print(set(list(data.flat)))

for i, (t, mask) in enumerate(zip(timestamps, spread_masks)):
    plt.clf()
    plt.imshow(mask * data + 100 * np.logical_not(mask), vmin=0, vmax=np.max(data))
    plt.title(f"Fire spread at {t}s")
    plt.tight_layout()
    plt.savefig(f'frames/{i:03}.png')