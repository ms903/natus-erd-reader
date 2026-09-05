"""Native decoder differential, quantization and buffer tests."""
import io
import random
import unittest
import numpy as np
from natus_erd._export_worker import native_available
from natus_erd.decoder import decode_schema9_packet
from ._fixture import _encode_packet, SHORTED

class NativePacketTests(unittest.TestCase):
    def setUp(self):
        if not native_available():
            self.skipTest('optional native extension unavailable')
        from natus_erd import _native
        self.native = _native

    def test_random_differential_packets_and_repeated_selections(self):
        rng = random.Random(20260905)
        for trial in range(80):
            count = rng.randrange(2,30)
            shorted = set(rng.sample(range(276),rng.randrange(0,15)))
            values = [[rng.randrange(-200000,200000) for _ in range(276)]]
            for _ in range(count-1):
                values.append([max(-2**31,min(2**31-1,v+rng.choice((-32768,-300,-128,-1,0,1,127,128,32767,100000)))) for v in values[-1]])
            payload = _encode_packet(values,shorted=shorted)
            selected = tuple(rng.randrange(276) for _ in range(20))+(0,0)
            start,stop = rng.randrange(count-1),count
            expected = decode_schema9_packet(io.BytesIO(payload),offset=0,byte_end=len(payload),sample_count=count,
                start=start,stop=stop,n_channels=276,shorted=tuple(i in shorted for i in range(276)),selected=selected)
            out = np.full((len(selected),stop-start),np.nan)
            self.native.process(payload,bytes(i in shorted for i in range(276)),selected,count,start,stop,1,(),out,stop-start,0)
            np.testing.assert_equal(out,expected)

    def test_invalid_payloads_and_output_buffers_rejected(self):
        values = [[10]*276,[11]*276]
        payload = _encode_packet(values)
        mask = bytes(i in SHORTED for i in range(276))
        for corrupt in (payload[:-1],payload+b'x',bytes([2])+payload[1:],payload[:40]):
            with self.assertRaises(ValueError):
                self.native.process(corrupt,mask,(0,),2,0,2,1,(),bytearray(16),2,0)
        for output in (bytes(16),bytearray(15),memoryview(bytearray(32))[::2]):
            with self.assertRaises((ValueError,BufferError)):
                self.native.process(payload,mask,(0,),2,0,2,1,(),output,2,0)
        for selected in ((-1,),(276,),(True,),()):
            with self.assertRaises(ValueError):
                self.native.process(payload,mask,selected,2,0,2,1,(),bytearray(16),2,0)

    def test_ties_to_even_quantization_and_integer_extremes(self):
        # q = native / 2: include both signs and even/odd neighbors.
        values = [[v]*276 for v in (-7,-5,-3,-1,1,3,5,7)]
        payload = _encode_packet(values)
        mask = bytes(i in SHORTED for i in range(276))
        output = bytearray(16)
        parameters = ((0,1.,-8.,8.,-4,4,0,1),)
        self.native.process(payload,mask,(0,),8,0,8,2,parameters,output,8,0)
        np.testing.assert_equal(np.frombuffer(output,dtype='<i2'),[-4,-2,-2,0,0,2,2,4])
        extreme = [[-2**31]*276,[2**31-1]*276]
        encoded = _encode_packet(extreme)
        decoded = np.empty((1,2))
        self.native.process(encoded,mask,(0,),2,0,2,1,(),decoded,2,0)
        np.testing.assert_equal(decoded,[[-2**31,2**31-1]])
        stats = self.native.process(encoded,mask,(0,),2,0,2,0,(),None,2,0)
        self.assertEqual(stats,((-2**31,2**31-1,2**32-1),))


