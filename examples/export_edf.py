from natus_erd import NatusERDReader, export_edf

reader = NatusERDReader.open(r"D:\data\recording")
result = export_edf(reader, r"D:\output.edf")
print(result.file_bytes, result.elapsed_seconds)
