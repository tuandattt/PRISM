#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import List

from mas_framework.tools.coding.executor_utils import function_with_timeout
from mas_framework.tools.coding.executor_types import ExecuteResult, Executor


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    
def execute_code_get_return(code: str, timeout: int = 10):
    """Run LLM-written code and read back its ``answer``.

    The code is MODEL-GENERATED, so it can loop forever or hand sympy a cubic it
    will grind on for minutes. This used to be a bare exec() with no timeout,
    called synchronously from an async agent -- so a single bad program blocked
    the entire event loop, freezing every concurrent query in the batch, and
    asyncio.wait_for could not fire because the loop had no chance to run its
    timer.

    ``get_output`` in this same module already had the timeout; this path was
    simply missed.

    A timed-out thread cannot be killed, so it keeps running in the background;
    PropagatingThread is a daemon so a leaked one does not stop the process from
    exiting.
    """
    def _run():
        local_vars = {}
        exec(code, {}, local_vars)
        return local_vars.get('answer')

    try:
        return function_with_timeout(_run, (), timeout)
    except TimeoutError:
        return f"Error occurred: code execution exceeded {timeout}s"
    except Exception as e:
        return f"Error occurred: {e}"

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                # A FRESH namespace per test, not globals().
                #
                # exec'ing into globals() leaks every definition a submission
                # makes into the next one. Measured on MBPP: an empty submission
                # fails 0/60 tasks in a clean process, then passes 59/60 after
                # the reference solutions have run -- the asserts resolve the
                # function name from the previous submission's leftovers.
                #
                # That is not only an evaluation bug. Cold start runs many
                # blueprints over the same tasks in ONE process, so a task
                # solved under Chain-2 would score as solved under every later
                # configuration whatever the agents produced, and those graphs
                # enter the training corpus labelled correct.
                #
                # A fresh dict still gets __builtins__ injected by exec, so
                # imports and builtins keep working.
                function_with_timeout(exec, (func_test_list[i], {}), timeout)
                success_tests.append(tests[i])
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False
        