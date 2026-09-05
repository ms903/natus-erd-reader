"""Run the distributed scripts with only their two local paths replaced."""
from pathlib import Path
import shutil
import unittest
import uuid
from ._fixture import build_continuous_recording


class ExamplesTests(unittest.TestCase):
    def test_read_stream_and_export_examples(self):
        root=Path.cwd()/('.natus-examples-test-'+uuid.uuid4().hex)
        root.mkdir()
        self.addCleanup(shutil.rmtree,root)
        fixture=build_continuous_recording(root,samples=128)
        loaded={}
        for name in ('read_window','streaming','export_edf'):
            path=Path(__file__).resolve().parents[1]/'examples'/(name+'.py')
            source=path.read_text().replace('D:\\data\\recording',str(fixture.directory)).replace('D:\\output.edf',str(root/'example.edf'))
            namespace={'__name__':'__main__','__file__':str(path)}
            exec(compile(source,str(path),'exec'),namespace)
            loaded[name]=namespace
        self.assertEqual(loaded['read_window']['data'].shape,(3,128))
        self.assertEqual((loaded['streaming']['samples'],loaded['streaming']['missing']),(128,0))
        result=loaded['export_edf']['result']
        self.assertEqual((result.logical_samples,result.channel_count),(128,276))
