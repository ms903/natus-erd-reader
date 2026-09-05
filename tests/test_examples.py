"""Exercise the short source-distribution examples with a synthetic recording."""
import importlib.util
from pathlib import Path
import shutil
import unittest
import uuid

from ._fixture import build_continuous_recording


class ExamplesTests(unittest.TestCase):
    def test_read_stream_and_export_examples(self):
        root = Path.cwd()/('.natus-examples-test-'+uuid.uuid4().hex)
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        fixture = build_continuous_recording(root, samples=128)
        loaded = {}
        for name in ('read_window', 'streaming', 'export_edf'):
            path = Path(__file__).resolve().parents[1]/'examples'/(name+'.py')
            spec = importlib.util.spec_from_file_location('example_'+name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[name] = module
        self.assertEqual(loaded['read_window'].read_window(fixture.directory).shape, (3, 128))
        self.assertEqual(loaded['streaming'].summarize(fixture.directory),
                         {'samples_per_channel': 128, 'nan_values': 0})
        result = loaded['export_edf'].convert(fixture.directory, root/'example.edf', 0, 128)
        self.assertEqual(result.logical_samples, 128)
        self.assertEqual(result.channel_count, 272)
