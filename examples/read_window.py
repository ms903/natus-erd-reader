"""Read a small window from one recording."""
from pathlib import Path
from natus_erd import NatusERDReader


def read_window(path):
    reader = NatusERDReader.open(path)
    data = reader.read_samples(0, min(2048, reader.info.n_samples), channels=[0, 1, 2])
    print(reader.info.sample_rate, data.shape)
    return data


if __name__ == "__main__":
    read_window(Path(r"D:\path\to\recording"))
