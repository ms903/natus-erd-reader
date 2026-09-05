"""Export an explicitly selected continuous interval to EDF+C."""
from pathlib import Path
from natus_erd import NatusERDReader, export_edf, plan_edf


def convert(path, output, start, stop):
    reader = NatusERDReader.open(path)
    print(tuple(reader.iter_stored_ranges()))
    plan = plan_edf(reader, start=start, stop=stop)
    print(plan.output_bytes, plan.backend, plan.shorted_channels)
    return export_edf(reader, output, start=start, stop=stop)


if __name__ == "__main__":
    convert(Path(r"D:\path\to\recording"), Path("window.edf"), 0, 2048)
