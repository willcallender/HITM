import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from os import listdir, remove

# Convenience imshow wrapper
def show(*args, **kwargs):
    # Swap the ymin and ymax values in extent, if it exists
    if 'extent' not in kwargs:
    #     extent = kwargs['extent']
    #     kwargs['extent'] = [extent[0], extent[1], extent[3], extent[2]]
    # else:
        # If extent doesn't exist, create it with flipped y-axis
        shape = args[0].shape
        kwargs['extent'] = [0, shape[1], 0, shape[0]]
    
    # Call imshow with modified arguments
    plt.imshow(*args, **kwargs)
    plt.show()

filename = 'firespreaddata/firstarrival.txt'

# Load the data, convert to row-major order to prevent confusion and headaches
firstArrivalTime = np.transpose(np.loadtxt(filename))
firstArrivalTime = firstArrivalTime - np.min(firstArrivalTime)
# print(set(list(data.flat)))
# print(firstArrivalTime)
print(firstArrivalTime.shape)
# show(firstArrivalTime)

def get_spread_mask(timestamps):
    spread_mask = np.zeros((len(timestamps), *firstArrivalTime.shape), dtype=bool)
    for i, timestamp in enumerate(timestamps):
        spread_mask[i, :] = firstArrivalTime <= timestamp
    return spread_mask

timestamps = np.arange(101) # np.linspace(0, 100, 201)
spread_masks = get_spread_mask(timestamps)
# print(spread_masks.shape) # (101, 380, 340)
# show(spread_masks[0], cmap='binary')

# Create a video of the fire spread
# for i, (t, mask) in enumerate(zip(timestamps, spread_masks)):
#     plt.clf()
#     show(mask * data + 100 * np.logical_not(mask), vmin=0, vmax=np.max(data))
#     plt.title(f"Fire spread at {t}s")
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f'frames/{i:03}.p

# use convolution to get the edge of the fire at each time step
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])
edges = np.zeros_like(spread_masks)
for i, (t, spread_mask) in enumerate(zip(timestamps, spread_masks)):
    edges[i] = (convolve2d(spread_mask, kernel, mode='same', boundary='fill', fillvalue=0) != 8) & spread_mask
# show(edges[50], cmap='binary')
# show(edge, cmap='hot')
# plt.show()

# get first ignition area
first_ignition = np.argwhere(firstArrivalTime == 0)
# test = firstArrivalTime == 0
# print(first_ignition)
# show(test)
# plt.show()

# centroid of first ignition area
first_ignition_centroid = np.mean(first_ignition, axis=0)
print(first_ignition_centroid)

# get distance from centroid to all points
distances = np.zeros_like(firstArrivalTime)
for x in range(firstArrivalTime.shape[0]):
    for y in range(firstArrivalTime.shape[1]):
        distances[x,y] = np.sqrt((x - first_ignition_centroid[0])**2 + \
            (y - first_ignition_centroid[1])**2)

# show(distance)


# get angle from centroid to each point
angles = np.zeros_like(firstArrivalTime)
for x in range(firstArrivalTime.shape[0]):
    for y in range(firstArrivalTime.shape[1]):
        angles[x,y] = np.arctan2(x - first_ignition_centroid[0], \
            y - first_ignition_centroid[1])
# show(angles)

# radially unwrap the distance to the edge at each time step
currentSpread = []
for i, (t, spread_mask, edge) in enumerate(zip(timestamps, spread_masks, edges)):
    # get the coordinates of the edge
    edgePoints = np.argwhere(edge)
    # get the distance to the edge at each point
    edgeDists = distances[edgePoints[:,0], edgePoints[:,1]]
    # get the angle at each point
    edgeAngles = angles[edgePoints[:,0], edgePoints[:,1]]
    
    # append the distance to the edge at each angle to currentSpread
    currentSpread.append((edgeAngles, edgeDists))

# delete all files from frames folder
for fname in listdir('frames'):
    remove(f'frames/{fname}')

# plot the spread at each time step
for i, spread in enumerate(currentSpread):
    plt.clf()
    plt.scatter(spread[0], spread[1], s=1)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, np.max(distances))
    plt.title(f"Fire spread at {i}s")
    plt.tight_layout()
    plt.savefig(f'frames/{i:03}.png')