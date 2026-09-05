"""Process one chunk at a time."""
from pathlib import Path
import numpy as np
from natus_erd import NatusERDReader


def summarize(path):
    reader = NatusERDReader.open(path)
    samples = missing = 0
    for chunk in reader.iter_samples(channels=[0, 1, 2], chunk_samples=2048):
        samples += chunk.shape[1]
        missing += int(np.isnan(chunk).sum())
    return {"samples_per_channel": samples, "nan_values": missing}


if __name__ == "__main__":
    print(summarize(Path(r"D:\path\to\recording")))
