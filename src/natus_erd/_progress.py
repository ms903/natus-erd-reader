"""Console progress for the three export passes."""
from contextlib import contextmanager


@contextmanager
def reporting(progress):
    if progress is False:
        yield None
        return
    if callable(progress):
        yield progress
        return
    if progress is not True:
        raise TypeError("progress must be True, False, or callable")
    from tqdm.auto import tqdm

    bar = None
    stage = None
    labels = {"range_scan": "Scanning", "write": "Writing", "verify": "Verifying"}

    def update(event):
        nonlocal bar, stage
        current = event["stage"]
        if current not in labels:
            return
        if current != stage:
            if bar is not None:
                bar.close()
            stage = current
            bar = tqdm(total=event["total"], desc=labels[current],
                       unit="sample" if current == "range_scan" else "record",
                       dynamic_ncols=True)
        completed = event.get("samples", event.get("records", 0))
        assert bar is not None
        bar.update(completed-bar.n)

    try:
        yield update
    finally:
        if bar is not None:
            bar.close()
