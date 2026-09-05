from natus_erd import NatusERDReader

reader = NatusERDReader.open(r"D:\data\recording")
data = reader.read_samples(0, min(2048, reader.info.n_samples), channels=[0, 1, 2])
print(reader.info.sample_rate, data.shape)
