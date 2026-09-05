"""EDF+D source-time mapping, boundary integrity and independent reading."""
from fractions import Fraction
from pathlib import Path
import shutil
from struct import pack, pack_into, unpack_from
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from natus_erd import NatusERDReader, Event, ReadLimits, export_edf, plan_edf
from natus_erd import DataIntegrityError, UnsupportedFormatError
from natus_erd._edf_codec import calibrate
from natus_erd._edf_verify import verify_export
from natus_erd._export_worker import combine_stats, ordered_work, native_available
from ._fixture import (build_discontinuous_recording, _annotation_layout,
                       _generic_header, _stc_entry, _snc)


class DiscontinuousExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd()/('.natus-edfd-test-'+uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fixture = build_discontinuous_recording(self.root)
        self.reader = NatusERDReader.open(self.fixture.directory)

    def export(self, **options):
        target = self.root/(uuid.uuid4().hex+'.edf')
        settings = dict(events='none', progress=False)
        settings.update(options)
        return target, export_edf(self.reader, target, **settings)

    def test_exact_onsets_counts_and_all_stored_waveforms(self):
        plan = plan_edf(self.reader, events='none')
        target, result = self.export()
        self.assertEqual(result.edf_format, 'EDF+D')
        self.assertEqual(result.stored_ranges, ((0,32),(544,576)))
        self.assertEqual((result.logical_samples,result.stored_samples),(576,64))
        self.assertEqual((result.stored_seconds,result.time_span_seconds,result.gap_seconds),
                         (Fraction(1,8),Fraction(9,8),Fraction(1)))
        payload = target.read_bytes()
        self.assertEqual(payload[192:197], b'EDF+D')
        offset, size, stride = _annotation_layout(payload)
        onsets = [Fraction(payload[i:i+size].split(b'\x14')[0].decode())
                  for i in range(offset,len(payload),stride)]
        expected = [Fraction(1,8)+Fraction(s,512) for a,b in ((0,32),(544,576))
                    for s in range(a,b,plan.record_samples)]
        self.assertEqual(onsets, expected)
        self.assertEqual(list(plan.record_starts()), [plan.record_sample(i) for i in range(plan.record_count)])
        with self.assertRaises(IndexError): plan.record_sample(plan.record_count)
        codes = np.ndarray((plan.record_count,276,plan.record_samples),dtype='<i2',
            buffer=payload,offset=int(payload[184:192]),strides=(stride,plan.record_samples*2,2))
        stored = codes.transpose(1,0,2).reshape(276,-1)
        np.testing.assert_array_equal(stored[249],32767)
        np.testing.assert_array_equal(stored[256],np.array(self.fixture.expected[256])[np.r_[0:32,544:576]])
        np.testing.assert_array_equal(stored[273:276],0)

    def test_gap_boundary_and_outside_events_keep_original_times(self):
        samples = (-16,0,16,32,256,544,560,576,600)
        self.reader._events = tuple(Event(1000+s,s,'event'+str(s)) for s in samples)
        for policy in ('full','types','none'):
            plan = plan_edf(self.reader,events=policy,annotation_bytes=52580)
            target, result = self.export(events=policy,annotation_bytes=52580)
            self.assertEqual(plan.record_samples,16)
            payload = target.read_bytes()
            start,size,stride = _annotation_layout(payload)
            slots = [payload[i:i+size] for i in range(start,len(payload),stride)]
            self.assertEqual(result.event_count,0 if policy=='none' else len(samples))
            if policy != 'full': continue
            for sample,index in zip(samples,(0,0,1,2,2,2,3,3,3)):
                onset=Fraction(1,8)+Fraction(sample,512)
                from natus_erd.edf_export import _tal
                self.assertIn(_tal(onset,'event'+str(sample)),slots[index])

    def test_windows_beginning_or_ending_in_gap(self):
        for first,last,ranges,kind in ((16,560,((16,32),(544,560)),'EDF+D'),
              (32,576,((544,576),),'EDF+C'),(0,544,((0,32),),'EDF+C'),
              (40,560,((544,560),),'EDF+C')):
            with self.subTest(window=(first,last)):
                plan=plan_edf(self.reader,start=first,stop=last,events='none')
                target,result=self.export(start=first,stop=last)
                self.assertEqual(result.stored_ranges,ranges)
                self.assertEqual(result.edf_format,kind)
                onset=plan._origin+Fraction(plan.record_sample(0))/plan.sample_rate
                self.assertGreaterEqual(onset,0)
                self.assertLess(onset,1)
                self.assertEqual(plan.logical_samples,last-first)
                self.assertTrue(target.exists())
        with self.assertRaisesRegex(UnsupportedFormatError,'No stored samples'):
            self.export(start=32,stop=544)

    def test_fractional_grid_and_unrepresentable_gap_onsets(self):
        for spans,success in ((((0,252),(502,753)),False),
                              (((0,251),(502,753)),True),
                              (((0,251),(503,754)),False)):
            root=self.root/uuid.uuid4().hex;root.mkdir()
            fixture=build_discontinuous_recording(root,spans=spans,sample_rate=125.5)
            reader=NatusERDReader.open(fixture.directory)
            target=root/'out.edf'
            if success:
                result=export_edf(reader,target,channels=[1],events='none',progress=False)
                self.assertEqual(result.stored_seconds,4)
                self.assertEqual(result.gap_seconds,2)
            else:
                with self.assertRaises(UnsupportedFormatError):
                    export_edf(reader,target,channels=[1],events='none',progress=False)
                self.assertFalse(target.exists())
                self.assertFalse(list(root.glob('*.partial-*')))

    def test_no_positive_duration_grid_for_single_sample_records(self):
        with self.assertRaisesRegex(UnsupportedFormatError,'No exact EDF record grid'):
            self.export(start=1)

    def test_small_reads_chunk_boundaries_and_progress(self):
        self.reader=NatusERDReader.open(self.fixture.directory,
            limits=ReadLimits(max_read_samples=1,max_read_bytes=8))
        updates=[]
        path,result=self.export(chunk_samples=32,progress=updates.append)
        self.assertEqual(result.stored_samples,64)
        scan=[u for u in updates if u['stage']=='range_scan']
        self.assertEqual(scan[-1]['samples'],64)
        self.assertTrue(all(u['total']==64 for u in scan))
        for stage in ('write','verify'):
            notices=[u for u in updates if u['stage']==stage]
            self.assertEqual(notices[-1]['records'],result.record_count)
        reference,_=self.export(backend='python',workers=1,chunk_samples=64)
        self.assertEqual(path.read_bytes(),reference.read_bytes())

    def test_twelve_hour_gap_does_not_allocate_or_write_missing_samples(self):
        shift=512*12*3600
        etc=self.fixture.first_erd.with_suffix('.etc')
        payload=bytearray(etc.read_bytes())
        for offset in range(352,len(payload),16):
            stamp=unpack_from('<i',payload,offset+4)[0]
            if stamp >= 1544: pack_into('<i',payload,offset+4,stamp+shift)
        etc.write_bytes(payload)
        self.fixture.stc.write_bytes(_generic_header(1)+pack('<ii12i',1,1,*([0]*12))
            +_stc_entry(self.fixture.stc.stem,1000,1575+shift,0,stored_samples=64))
        _snc(self.fixture.stc.with_suffix('.snc'),512,1575+shift)
        self.reader=NatusERDReader.open(self.fixture.directory)
        with patch.object(self.reader,'read_samples',wraps=self.reader.read_samples) as reads:
            path,result=self.export(channels=[1,256],chunk_samples=32)
        self.assertEqual(result.stored_samples,64)
        self.assertEqual(result.gap_seconds,12*3600+1)
        self.assertLess(path.stat().st_size,4096)
        self.assertTrue(all(call.args[1]-call.args[0] <=16 for call in reads.call_args_list))

    def test_native_and_python_equal_across_packet_carries_and_gaps(self):
        if not native_available(): self.skipTest('optional native extension unavailable')
        outputs=[]
        for backend,workers in (('python',1),('native',1),('native','auto')):
            path,_=self.export(backend=backend,workers=workers,chunk_samples=32)
            outputs.append(path.read_bytes())
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[1],outputs[2])

    def test_second_segment_tal_waveform_and_truncation_rejected(self):
        plan=plan_edf(self.reader,events='none',chunk_samples=32)
        target,_=self.export(chunk_samples=32)
        stats=None
        for _,_,part,_,_ in ordered_work(self.reader,plan): stats=combine_stats(stats,part)
        calibrations=tuple(calibrate(self.reader.channels[c],v,.5) for c,v in zip(plan.channels,stats))
        original=target.read_bytes()
        annotation,size,stride=_annotation_layout(original)
        second=plan._record_offsets[1]
        mutations=[]
        changed=bytearray(original)
        at=annotation+second*stride
        changed[at:at+size]=b'+0.125\x14\x14\0'.ljust(size,b'\0')
        mutations.append((changed,'annotation'))
        changed=bytearray(original)
        pack_into('<h',changed,int(original[184:192])+second*stride+256*plan.record_samples*2,0)
        mutations.append((changed,'waveform'))
        changed=bytearray(original)
        pack_into('<h',changed,int(original[184:192])+second*stride+249*plan.record_samples*2,0)
        mutations.append((changed,'shorted'))
        mutations.append((original[:-2],'length'))
        for changed,error in mutations:
            target.write_bytes(changed)
            with self.assertRaisesRegex(DataIntegrityError,error):
                verify_export(self.reader,target,plan,calibrations,.5)

    def test_cancel_failure_and_existing_output_cleanup(self):
        target=self.root/'cancel.edf'
        def cancel(event):
            if event['stage']=='verify': raise RuntimeError('cancel verification')
        with self.assertRaisesRegex(RuntimeError,'cancel verification'):
            export_edf(self.reader,target,events='none',progress=cancel)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob('*.partial-*')))
        target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            export_edf(self.reader,target,events='none',progress=False)
        self.assertEqual(target.read_bytes(),b'keep')

    def test_independent_edf_reader_restores_gap_and_all_eeg_values(self):
        try: from edf_reader import EdfWrapper
        except ImportError: self.skipTest('edf-reader interoperability extra not installed')
        target,_=self.export()
        external=EdfWrapper(str(target))
        self.addCleanup(external.close)
        self.assertTrue(external.header['is_edf_plus_d'])
        gaps=external.get_discontinuities()
        self.assertEqual(len(gaps),1)
        self.assertEqual(gaps[0][1]-gaps[0][0],1_000_000)
        info=external.read_ts_channel_basic_info()[1]
        self.assertEqual(info['fsamp'],512)
        self.assertEqual(info['nsamp'],64)
        self.assertEqual(info['end_time']-info['start_time'],1_125_000)
        values=external.read_ts_channels_uutc(['CH001'],[None,None])
        expected=self.reader.read_samples(0,576,[1])
        self.assertEqual(values.shape,(1,576))
        np.testing.assert_allclose(values,expected,atol=.5,rtol=0,equal_nan=True)
