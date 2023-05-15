from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np

class MultiThreadedExecutor:
    debug = True
    def __init__(self, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)

    def fill(self, data_func, in_data, out_shape = None):
        def _fill(data_func, in_data, out_data, first, last):
            out_data[first:last] = [data_func(d) for d in in_data[first:last]]

        futures = {}
        out_data = np.zeros(out_shape or len(in_data))
        step = np.ceil(len(in_data) / self.threads).astype(np.int_)
        for i in range(min(self.threads, len(in_data))):
            args = (_fill,
                    data_func,
                    in_data,
                    out_data,
                    i * step,
                    min((i + 1) * step, len(in_data)))
            if self.debug:
                #execute single threaded to debug
                _fill(*args[1:])
            else:
                futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        return out_data
    
    def fill_list(self, data_func, in_data):
        def _fill(data_func, in_data, out_data, first, last):
            for i, d in enumerate(in_data[first:last]):
                out_data[first+i] = data_func(d)

        futures = {}
        out_data = [None for i in range(len(in_data))]
        step = np.ceil(len(in_data) / self.threads).astype(np.int_)
        for i in range(min(self.threads, len(in_data))):
            args = (_fill,
                    data_func,
                    in_data,
                    out_data,
                    i * step,
                    min((i + 1) * step, len(in_data)))
            if self.debug:
                #execute single threaded to debug
                _fill(*args[1:])
            else:
                futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        return out_data
    
    def exec(self, data_func, in_data):
        def _exec(data_func, in_data, first, last):
            for d in in_data[first:last]:
                data_func(d)

        futures = {}
        step = np.ceil(len(in_data) / self.threads).astype(np.int_)
        for i in range(min(self.threads, len(in_data))):
            args = (_exec,
                    data_func,
                    in_data,
                    i * step,
                    min((i + 1) * step, len(in_data)))
            if self.debug:
                #execute single threaded to debug
                _exec(*args[1:])
            else:
                futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.ALL_COMPLETED)

    def __del__(self):
        self.executor.shutdown(False)