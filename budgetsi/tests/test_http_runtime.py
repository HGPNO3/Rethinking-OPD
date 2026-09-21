"""CPU transport tests; no model imports or production endpoints."""
import ast
import json
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from budgetsi.http_runtime import BurstHTTPServer


class BurstTests(unittest.TestCase):
    def test_backlog_only_policy_change(self):
        self.assertEqual(BurstHTTPServer.request_queue_size, 128)
        self.assertIs(BurstHTTPServer.process_request, ThreadingHTTPServer.process_request)

    def load_factory(self, filename, name):
        # Run actual factory AST without importing model/GPU dependencies.
        tree = ast.parse((Path(__file__).parents[1] / filename).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = dict(BaseHTTPRequestHandler=BaseHTTPRequestHandler, HTTPServer=HTTPServer,
                         ThreadingHTTPServer=ThreadingHTTPServer, threading=threading, json=json,
                         traceback=SimpleNamespace(print_exc=lambda: None))
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
        return namespace[name]

    def test_real_factories_parallel_burst_and_error_contract(self):
        for filename, name, endpoint in [('remote_teacher.py', 'serve', '/'), ('social_loop.py', 'start_server', '/v1/completions')]:
            with self.subTest(factory=name):
                def call(data):
                    if data.get('fail'): raise ValueError('sentinel')
                    return {'id': data['id']}
                engine = SimpleNamespace(parallel=True, call=call)
                factory = self.load_factory(filename, name)
                server = factory(SimpleNamespace(engine=engine, dispatch=call), 0) if name == 'serve' else factory(engine)
                self.assertIsInstance(server, BurstHTTPServer)
                if name == 'serve': threading.Thread(target=server.serve_forever, daemon=True).start()
                url = 'http://127.0.0.1:%s%s' % (server.server_port, endpoint)
                def post(value):
                    with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(value).encode(), headers={'Content-Type': 'application/json'}), timeout=5) as response:
                        return json.load(response)
                try:
                    barrier = threading.Barrier(64)
                    def request(i):
                        barrier.wait(timeout=5)
                        return post({'id': i})
                    with ThreadPoolExecutor(max_workers=64) as pool:
                        results = list(pool.map(request, range(64)))
                    self.assertEqual(results, [{'id': i} for i in range(64)])
                    with self.assertRaises(urllib.error.HTTPError) as error: post({'fail': True})
                    self.assertEqual(error.exception.code, 500)
                    self.assertIn('sentinel', error.exception.read().decode())
                finally:
                    server.shutdown(); server.server_close()

    def test_serial_mode_stays_serial(self):
        for filename, name in [('remote_teacher.py', 'serve'), ('social_loop.py', 'start_server')]:
            engine = SimpleNamespace(parallel=False)
            factory = self.load_factory(filename, name)
            server = factory(SimpleNamespace(engine=engine), 0) if name == 'serve' else factory(engine)
            try:
                self.assertIs(type(server), HTTPServer)
            finally:
                if name == 'start_server': server.shutdown()
                server.server_close()


if __name__ == '__main__': unittest.main()
