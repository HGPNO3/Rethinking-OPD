"""Loopback HTTP transport for concurrent model requests.

The accept backlog buffers connection establishment only. Model concurrency,
request validation, numerical settings and error handling stay with the callers.
"""
from http.server import ThreadingHTTPServer


class BurstHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128
