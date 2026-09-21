"""Actual HTTP envelope + fake backend lifecycle, independently of GPU loading."""
import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from budgetsi.phased_teacher import PhasedTeacher
from budgetsi.remote_teacher import RemoteTeacher,serve


class FakeGenerator:
    def __init__(self,*a,**k):self.process=SimpleNamespace(poll=lambda:None)
    def start(self):pass
    def close(self):pass
    def call(self,data):return {'usage':{'prompt_tokens':2,'completion_tokens':1}}


class PhasedTransportTests(unittest.TestCase):
    def test_real_http_timing_and_three_member_barrier(self):
        with tempfile.TemporaryDirectory() as tmp,patch('budgetsi.phased_teacher.VLLMTeacher',FakeGenerator):
            bindings=[{'variant':i} for i in range(3)]
            service=PhasedTeacher([{}]*3,bindings,[],[],{},Path(tmp))
            server=serve(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                clients=[RemoteTeacher(f'http://127.0.0.1:{server.server_port}/',b) for b in bindings]
                with concurrent.futures.ThreadPoolExecutor(3) as pool:
                    rows=list(pool.map(lambda c:c.call('collect',{'model':'teacher'}),clients))
                self.assertEqual(len(rows),3)
                for c in clients:
                    self.assertGreaterEqual(c.timings[-1]['backend_roundtrip_seconds'],0)
                    self.assertGreaterEqual(c.timings[-1]['admission_queue_seconds'],0)
                    self.assertTrue(c.timings[-1]['id'])
                modes=[]
                def switch(mode):
                    self.assertTrue(service.scheduler.idle());modes.append(mode);service.mode=mode
                with patch.object(service.cohort,'switch',switch):
                    with concurrent.futures.ThreadPoolExecutor(3) as pool:
                        list(pool.map(lambda c:c.call('phase',{'action':'collected'}),clients))
                        list(pool.map(lambda c:c.call('phase',{'action':'updated','done':True}),clients))
                    self.assertEqual(modes,['hf','closed'])
            finally:server.shutdown();server.server_close();service.close();thread.join()

if __name__=='__main__':unittest.main()
