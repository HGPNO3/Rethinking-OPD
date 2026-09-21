"""Read-only GPU samples for measured benchmark windows, no device allocation."""
import subprocess
import threading
import time


class GPUSampler:
    def __init__(self):
        self.rows=[];self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.thread.start()
    def run(self):
        while not self.stop_event.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                                             '--format=csv,noheader,nounits'],text=True,timeout=3)
                self.rows.append(dict(time=time.time(),gpus=[list(map(int,s.split(','))) for s in raw.splitlines()]))
            except (OSError,ValueError,subprocess.SubprocessError):
                self.rows.append(dict(time=time.time(),error='GPU sample unavailable'))
            self.stop_event.wait(1)
    def close(self):self.stop_event.set();self.thread.join()
