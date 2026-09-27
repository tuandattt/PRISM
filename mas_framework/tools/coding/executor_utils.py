#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from threading import Thread

def timeout_handler(_, __):
    raise TimeoutError()


def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        # A join() that TIMED OUT leaves .ret unset, and returning it raised
        # AttributeError -- which callers reported as the failure reason,
        # masking the real one ("code execution exceeded Ns"). Only .exc and
        # .ret set by a finished run() are meaningful here.
        super(PropagatingThread, self).join(timeout)
        exc = getattr(self, 'exc', None)
        if exc:
            raise exc
        return getattr(self, 'ret', None)
    

def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    # A thread running runaway model-generated code cannot be killed. Daemon so
    # a leaked one does not keep the interpreter alive at exit.
    thread = PropagatingThread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]
    

