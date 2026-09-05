"""Continuous EDF+C interoperability, resource and failure regressions."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
from struct import pack, pack_into
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from natus_erd import (NatusERDReader, ReadLimits, Event, export_edf, plan_edf,
                       DataIntegrityError, ResourceLimitError, UnsupportedFormatError)
from natus_erd._edf_codec import calibrate, header_labels, official_calibration
from natus_erd._export_worker import native_available, execution
from natus_erd._edf_verify import verify_export
from ._fixture import (build_recording, build_continuous_recording, _snc,
                       _generic_header, _annotation_layout, BASE_DATETIME)


class EdfExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd()/('.natus-export-test-'+uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fixture = build_continuous_recording(self.root)
        self.reader = NatusERDReader.open(self.fixture.directory)

    def export(self, **kwargs):
        target = self.root/(uuid.uuid4().hex+'.edf')
        options = dict(channels=[1, 2, 249, 256, 273], progress=False)
        options.update(kwargs)
        return target, export_edf(self.reader, target, **options)

    def test_defaults_and_small_reader_limits(self):
        limits = ReadLimits(max_read_samples=1, max_read_bytes=8)
        reader = NatusERDReader.open(self.fixture.directory, limits=limits)
        plan = plan_edf(reader)
        self.assertEqual(plan.channels, tuple(range(276)))
        self.assertEqual(plan.shorted_channels, (249, 251, 253, 255))
        self.assertGreater(plan.chunk_samples, limits.max_read_samples)
        self.assertIs(reader.limits, limits)
        result = export_edf(reader, self.root/'all.edf')
        self.assertEqual(result.channel_count, 276)
        self.assertEqual(result.shorted_channels, plan.shorted_channels)
        with self.assertRaises(ResourceLimitError):
            reader.read_samples(0, 2, [0])

    def test_progress_defaults_disable_callback_and_phase_timings(self):
        from contextlib import redirect_stderr
        from io import StringIO
        for setting in (True, False):
            output = StringIO()
            with redirect_stderr(output):
                _, result = self.export(progress=setting)
            for stage in ('Scanning', 'Writing', 'Verifying'):
                self.assertEqual(stage in output.getvalue(), setting)
            self.assertGreaterEqual(result.elapsed_seconds, result.scan_seconds+result.write_seconds+result.verify_seconds)
            self.assertGreater(result.verify_seconds, 0)
        updates=[]
        self.export(progress=updates.append)
        for stage in ('range_scan','write','verify'):
            events=[e for e in updates if e['stage']==stage]
            key='samples' if stage=='range_scan' else 'records'
            self.assertEqual(events[0][key],0)
            self.assertEqual(events[-1][key],events[-1]['total'])
            self.assertEqual([e[key] for e in events], sorted(e[key] for e in events))
        with self.assertRaises(TypeError):
            self.export(progress='yes')
        path, _ = self.export()
        self.assertEqual(path.read_bytes()[176:184],b'20.00.00')
        plan=plan_edf(self.reader)
        self.assertEqual(plan._utc_offset_seconds,28800)
        self.assertEqual(plan.channel_units,('uV',)*273+('%','bpm','uV'))

    def test_batched_readback_and_late_tal_and_truncation_corruption(self):
        from natus_erd._edf_write import ordered_work
        from natus_erd._export_worker import combine_stats
        plan=plan_edf(self.reader,channels=[1,2,249,256,273],chunk_samples=256,annotation_bytes=60000)
        path,_=self.export(chunk_samples=256,annotation_bytes=60000)
        stats=None
        for _,_,part,_,_ in ordered_work(self.reader,plan):
            stats=combine_stats(stats,part)
        calibrations=tuple(calibrate(self.reader.channels[c],v,.5) for c,v in zip(plan.channels,stats))
        with patch.object(self.reader,'read_samples',wraps=self.reader.read_samples) as reads:
            verify_export(self.reader,path,plan,calibrations,.5)
        self.assertEqual(reads.call_count,5)
        original=path.read_bytes()
        changed=bytearray(original)
        changed[-1]=1
        path.write_bytes(changed)
        with self.assertRaisesRegex(DataIntegrityError,'annotation'):
            verify_export(self.reader,path,plan,calibrations,.5)
        changed=bytearray(original)
        record_bytes=len(plan.channels)*plan.record_samples*2+plan.annotation_bytes
        late_code=256*(len(plan.channels)+2)+(plan.record_count-1)*record_bytes+4*plan.record_samples*2
        pack_into('<h',changed,late_code,-1)
        path.write_bytes(changed)
        with self.assertRaisesRegex(DataIntegrityError,'digital range'):
            verify_export(self.reader,path,plan,calibrations,.5)
        path.write_bytes(original[:-2])
        with self.assertRaisesRegex(DataIntegrityError,'length'):
            verify_export(self.reader,path,plan,calibrations,.5)

    def test_shorted_selection_uses_flags_and_preserves_positions(self):
        _, result=self.export(channels=[249,251])
        self.assertEqual(result.channel_count,2)
        self.assertEqual(result.shorted_channels,(249,251))
        root=self.root/'different-shorted'
        root.mkdir()
        fixture=build_continuous_recording(root,shorted={1,256})
        reader=NatusERDReader.open(fixture.directory)
        plan=plan_edf(reader,channels=[256,2,1])
        result=export_edf(reader,self.root/'different-shorted.edf',channels=[256,2,1],progress=False)
        self.assertEqual(result.shorted_channels,(256,1))
        self.assertEqual(plan.channels,(256,2,1))
        self.assertEqual(plan.shorted_channels,(256,1))

    def test_clock_and_events_read_once_per_plan(self):
        with patch.object(self.reader, 'read_clock', wraps=self.reader.read_clock) as clock:
            with patch.object(self.reader, 'read_events', wraps=self.reader.read_events) as events:
                plan_edf(self.reader, channels=[1])
        self.assertEqual(clock.call_count, 1)
        self.assertEqual(events.call_count, 1)

    def test_empty_gap_window_refused_before_decoding(self):
        root = self.root/'gaps'
        root.mkdir()
        fixture = build_recording(root)
        reader = NatusERDReader.open(fixture.directory)
        with patch.object(reader, 'read_clock', side_effect=AssertionError('premature clock read')):
            with self.assertRaisesRegex(UnsupportedFormatError, 'No stored samples'):
                export_edf(reader, self.root/'gap.edf', start=5, stop=6)
        self.assertFalse((self.root/'gap.edf').exists())

    def test_inexact_tail_and_fractional_rate_alignment(self):
        root = self.root/'fractional'
        root.mkdir()
        fixture = build_continuous_recording(root, sample_rate=125.5, samples=503)
        reader = NatusERDReader.open(fixture.directory)
        with self.assertRaisesRegex(UnsupportedFormatError, 'candidate aligned window start=0, stop=502'):
            export_edf(reader, self.root/'tail.edf', events='none', channels=[1])
        plan = plan_edf(reader, stop=502, events='none', channels=[1])
        self.assertEqual(plan.record_samples*plan.record_count, 502)
        self.assertEqual(Fraction(plan.record_duration_text), plan.record_samples/Fraction(251, 2))
        self.assertFalse((self.root/'tail.edf').exists())

    def test_labels_preserve_legal_names_and_map_unique_aliases(self):
        names = ("B'12", '中文', 'A'*20, 'same', 'same', 'NATUS0001', 'EDF Annotations')
        labels = header_labels(names)
        self.assertEqual(labels[0], "B'12")
        self.assertEqual(labels[5], 'NATUS0001')
        self.assertEqual(len(set(labels)), len(labels))
        self.assertTrue(all(len(label.encode('ascii')) <= 16 for label in labels))
        self.reader._channels = tuple(replace(c, name=names[c.index]) if c.index < len(names) else c
                                      for c in self.reader.channels)
        path, result = self.export(channels=list(range(len(names))))
        self.assertEqual(result.channel_labels, tuple(zip(range(len(names)), names, labels)))
        self.assertEqual(path.read_bytes()[256:272].rstrip(), b"B'12")

    def test_all_source_events_are_stored_without_window_filtering(self):
        original = (Event(999, -1, 'before\\\x00'), Event(1001, 1, '注释\x14\x15'),
                    Event(3000, 2000, 'after', note_type=3))
        self.reader._events = original
        for policy in ('full', 'types', 'none'):
            with self.subTest(policy=policy):
                path, result = self.export(start=512, stop=768, events=policy)
                payload = path.read_bytes()
                offset, size, stride = _annotation_layout(payload)
                annotations = b''.join(payload[a:a+size] for a in range(offset, len(payload), stride))
                self.assertEqual(result.event_count, 0 if policy == 'none' else 3)
                if policy == 'full':
                    self.assertIn(b'before\\\\\\u0000', annotations)
                    self.assertIn('注释\\u0014\\u0015'.encode(), annotations)
                    self.assertIn(b'after', annotations)
                    # Source stamp 999 stays at -513/512 seconds from window start.
                    from natus_erd.edf_export import _tal
                    plan = plan_edf(self.reader, start=512, stop=768, channels=[1])
                    self.assertIn(_tal(plan._origin+Fraction(-513, 512), 'before\\\\\\u0000'), annotations)
                elif policy == 'types':
                    self.assertEqual(annotations.count(b'type:'), 3)
                else:
                    self.assertNotIn(b'after', annotations)

    def test_dense_event_capacity_is_not_truncated(self):
        self.reader._events = tuple(Event(1001, 1, '界'*400) for _ in range(5))
        with self.assertRaises(ResourceLimitError):
            self.export(annotation_bytes=224)
        path, result = self.export()
        self.assertEqual(result.event_count, 5)
        self.assertEqual(path.read_bytes().count(('界'*400).encode()), 5)
        self.reader._events = (Event(1001, 1, 'X'*65000),)
        with self.assertRaises(ResourceLimitError):
            self.export()

    def test_auto_fallback_large_output_and_explicit_budgets(self):
        with patch('natus_erd._export_worker.native_available', return_value=False):
            config = execution(self.reader, (0,), 16, 224, 2*1024**3, 'auto', 'auto', 32*1024**2, None)
            self.assertEqual((config.backend, config.workers), ('python', 1))
            self.assertLessEqual(config.reserved_bytes, 32*1024**2)
            with self.assertRaises(UnsupportedFormatError):
                self.export(backend='native')
        with self.assertRaises(ValueError):
            self.export(backend='python', workers=2)
        with self.assertRaises(ResourceLimitError):
            self.export(max_output_bytes=100)
        with self.assertRaises(ResourceLimitError):
            self.export(memory_budget_bytes=1)

    def test_large_file_plan_has_no_default_output_ceiling(self):
        from natus_erd import SNCClock, ClockAnchor
        from ._fixture import BASE_TICKS
        count = 50_000_000
        self.reader._info = replace(self.reader.info, n_samples=count, end_stamp=999+count)
        clock = SNCClock(512., 1000, 999+count,
                         (ClockAnchor(1000, BASE_TICKS),
                          ClockAnchor(999+count, BASE_TICKS+round(Fraction(count-1, 512)*10_000_000))))
        with patch.object(self.reader, 'read_clock', return_value=clock):
            with patch.object(self.reader, 'iter_stored_ranges', return_value=iter(((0, count),))):
                plan = plan_edf(self.reader, channels=[1], events='none', backend='python')
        self.assertGreater(plan.output_bytes, 64*1024**2)
        self.assertEqual(plan.record_count*plan.record_samples, count)

    def test_named_timezone_with_optional_data(self):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo('Asia/Shanghai')
        except ZoneInfoNotFoundError:
            self.skipTest('install the timezones extra for named-zone checks')
        path, _ = self.export(timezone='Asia/Shanghai')
        self.assertEqual(path.read_bytes()[176:184], b'20.00.00')

    def test_source_directory_output_and_existing_target(self):
        target = self.fixture.directory/'new.edf'
        result = export_edf(self.reader, target, channels=[1])
        self.assertEqual(target.stat().st_size, result.file_bytes)
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            export_edf(self.reader, target, channels=[1])
        self.assertEqual(target.read_bytes(), original)

    def test_racing_destination_and_cancellation_cleanup(self):
        target = self.root/'race.edf'
        def race(event):
            if event['stage'] == 'publishing':
                target.write_bytes(b'other writer')
        with self.assertRaises(FileExistsError):
            export_edf(self.reader, target, channels=[1], progress=race)
        self.assertEqual(target.read_bytes(), b'other writer')
        def cancel(event):
            if event['stage'] == 'publishing':
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.export(progress=cancel)
        self.assertFalse(list(self.root.glob('*.partial-*')))

    def test_bad_codes_and_changed_waveform_fail_readback(self):
        for channel, corrupt, message in ((273,-1,'digital range'), (256,-32767,'source waveform'), (249,0,'shorted')):
            def mutate(reader, path, plan, calibrations, limit, progress=None):
                payload = bytearray(path.read_bytes())
                pack_into('<h', payload, int(payload[184:192]), corrupt)
                path.write_bytes(payload)
                verify_export(reader, path, plan, calibrations, limit, progress)
            with patch('natus_erd._edf_verify.verify_export', side_effect=mutate):
                with self.assertRaisesRegex(DataIntegrityError, message):
                    self.export(channels=[channel])
        self.assertFalse(list(self.root.glob('*.edf')))
        self.assertFalse(list(self.root.glob('*.partial-*')))

    def test_unrepresentable_quantization_fails_before_creating_partial(self):
        with patch('natus_erd._edf_write.calibrate', side_effect=UnsupportedFormatError('range')):
            with self.assertRaises(UnsupportedFormatError):
                self.export()
        self.assertFalse(list(self.root.glob('*.edf*')))
        channel = self.reader.channels[1]
        with self.assertRaises(UnsupportedFormatError):
            calibrate(channel, (-2**31, 2**31-1), .5)

    def test_mne_and_pyedflib_independent_readers(self):
        environment = patch.dict(os.environ, {'_MNE_FAKE_HOME_DIR': str(self.root),
                                              'MPLCONFIGDIR': str(self.root/'matplotlib')})
        environment.start()
        self.addCleanup(environment.stop)
        try:
            import mne
            import pyedflib
        except ImportError:
            self.skipTest('install the interop extra to run external-reader checks')
        self.reader._events = (Event(999, -1, 'before'), Event(1512, 512, 'inside'),
                               Event(3000, 2000, 'after'))
        selected = [1, 2, 249, 256, 272, 273, 274, 275]
        path, result = self.export(channels=selected)
        expected = self.reader.read_samples(0, 1024, selected, units='digital')
        expected[:2] *= self.reader.channels[1].scale_uv_per_count
        expected[2] = -8711
        expected[3:5] = 5151600-(expected[3:5]+32768)*(10303200/65535)
        expected[5:7] = 0
        expected[7] = 4.29e9+32768*(32767-4.29e9)/65535
        from datetime import timedelta
        header_time = BASE_DATETIME+timedelta(hours=8)
        labels = [self.reader.channels[c].name for c in selected]
        units = ['uV','uV','uV','uV','uV','%','bpm','uV']
        with pyedflib.EdfReader(str(path)) as edf:
            self.assertEqual(edf.signals_in_file, 8)
            self.assertEqual(edf.getSignalLabels(), labels)
            np.testing.assert_equal(edf.getNSamples(), [1024]*8)
            np.testing.assert_equal(edf.getSampleFrequencies(), [512]*8)
            self.assertEqual(edf.getStartdatetime().replace(microsecond=0),
                             header_time.replace(tzinfo=None, microsecond=0))
            self.assertEqual(edf.starttime_subsecond, BASE_DATETIME.microsecond*10)
            self.assertEqual([edf.getPhysicalDimension(i) for i in range(8)], units)
            for row in range(8):
                np.testing.assert_allclose(edf.readSignal(row), expected[row], atol=.5 if row < 2 else 1e-6, rtol=0)
            self.assertEqual(list(edf.readAnnotations()[2]), ['before', 'inside', 'after'])
            np.testing.assert_equal(edf.readSignal(2,digital=True),32767)
            np.testing.assert_equal(edf.readSignal(5,digital=True),0)
            np.testing.assert_equal(edf.readSignal(7,digital=True),0)
        raw = mne.io.read_raw_edf(str(path), preload=False, misc=labels[3:], verbose='ERROR')
        self.assertEqual(raw.ch_names, labels)
        self.assertEqual(raw.n_times, 1024)
        self.assertEqual(raw.info['sfreq'], 512)
        self.assertEqual(raw.info['meas_date'], header_time.replace(microsecond=0))
        values = raw.get_data()
        for row, unit in enumerate(result.channel_units):
            if unit == 'uV':
                values[row] *= 1e6
        np.testing.assert_allclose(values, expected, atol=.5, rtol=0)
        self.assertIn('inside', raw.annotations.description)
        inside = list(raw.annotations.description).index('inside')
        self.assertEqual(raw.annotations.onset[inside], 1.0)
        self.assertTrue(set(raw.annotations.description).issubset({'before', 'inside', 'after'}))
        self.assertEqual(result.event_count, 3)

    def test_official_auxiliary_calibration_and_verified_missing_codes(self):
        expected = [(256, (-32768,32767), 'uV', '5151600', '-5151600', -32768),
                    (273, (131070,131070), '%', '0', '102.3', 131070),
                    (274, (131070,131070), 'bpm', '0', '1023', 131070),
                    (275, (0,0), 'uV', '4.29e+09', '32767', -32768)]
        for index, stats, unit, pmin, pmax, raw_min in expected:
            cal = calibrate(self.reader.channels[index],stats,.5)
            self.assertEqual((cal.unit,cal.pmin,cal.pmax,cal.raw_min),(unit,pmin,pmax,raw_min))
        for index, stats in ((256,(-32769,0)),(273,(98,131070)),(274,(60,60)),(275,(1,1))):
            with self.assertRaisesRegex(UnsupportedFormatError, str(index)):
                calibrate(self.reader.channels[index],stats,.5)

    def test_narrow_channels_include_annotation_bytes_in_memory_budget(self):
        config = execution(self.reader,(0,),4,8192,10**7,'python',1,128*1024**2,None)
        encoded = (config.chunk_samples//4)*(8192+8)
        self.assertLessEqual(encoded,16*1024**2)
        self.assertLessEqual(config.reserved_bytes,128*1024**2)
        with self.assertRaises(ResourceLimitError):
            execution(self.reader,(0,),4,8192,10**7,'python',1,128*1024**2,1_000_000)

    def test_disk_failure_worker_failure_and_publication_failure_leave_no_output(self):
        with patch('natus_erd._edf_write.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(ResourceLimitError):
                self.export()
        from natus_erd._edf_write import ordered_work
        def fail_second(reader,plan,calibrations=None):
            if calibrations is not None:
                raise DataIntegrityError('synthetic worker failure')
            yield from ordered_work(reader,plan)
        with patch('natus_erd._edf_write.ordered_work',side_effect=fail_second):
            with self.assertRaises(DataIntegrityError):
                self.export()
        with patch('os.fsync',side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(OSError):
                self.export()
        self.assertFalse(list(self.root.glob('*.edf')))
        self.assertFalse(list(self.root.glob('*.partial-*')))

    def test_last_progress_callback_source_change_refuses_publication(self):
        def changed(event):
            if event['stage']=='publishing':
                self.fixture.stc.with_suffix('.snc').write_bytes(b'synthetic late change')
        with self.assertRaises(DataIntegrityError):
            self.export(progress=changed)
        self.assertFalse(list(self.root.glob('*.edf*')))

    def test_worker_count_and_backend_byte_equivalence(self):
        if not native_available():
            self.skipTest('optional native extension unavailable')
        expected = None
        for backend,workers in (('python',1),('native',1),('native',2),('native',4)):
            path,_ = self.export(channels=[1,2,249,256,272,273,274,275],backend=backend,workers=workers,chunk_samples=256,annotation_bytes=60000)
            data = path.read_bytes()
            if expected is None:
                expected = data
            self.assertEqual(data,expected)

    def test_expanded_vendor_label_layout_and_unknown_fallback(self):
        names = [f'C{i}' for i in range(512)]+[f'DC{i}' for i in range(1,17)]+['TRIG','OSAT','PR','Pleth']
        def write_names(values):
            text = '(.(."ChanNames", ('+', '.join(json.dumps(n) for n in values)+')))'
            payload = text.encode()+b'\0\0'
            self.fixture.stc.with_suffix('.ent').write_bytes(_generic_header(3)+pack('<4i',2,len(payload)+16,0,0)+payload+bytes(16))
        write_names(names)
        reader = NatusERDReader.open(self.fixture.directory)
        self.assertEqual([c.name for c in reader.channels[256:]],names[512:])
        write_names(names[:-1])
        reader = NatusERDReader.open(self.fixture.directory)
        self.assertTrue(all(not c.name_resolved for c in reader.channels[256:]))
