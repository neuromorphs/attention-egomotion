import numpy as np

labelled_events = np.load('../dataset/dataset-sample/dataset-sample/recordings/2024-01-17T12-03-38.264Z/labelled_events.npy')

time = np.unique(labelled_events['t'])

a = np.zeros((400, 400))

print()
