import numpy as np
from natus_erd import NatusERDReader

reader = NatusERDReader.open(r"D:\data\recording")
samples = missing = 0
for chunk in reader.iter_samples(channels=[0, 1, 2], chunk_samples=2048):
    samples += chunk.shape[1]
    missing += int(np.isnan(chunk).sum())
print("samples per channel:", samples, "NaNs:", missing)
