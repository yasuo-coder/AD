from pathlib import Path
"""Run with stdlib unittest; no additional test dependency is required."""
import importlib
import inspect
import time
import unittest
import torch
from BrainPromptPro.runtime import ROOT,output_path,write_json


def main():
    torch.set_num_threads(2)
    suite=unittest.TestSuite()
    for path in sorted(Path(__file__).parent.glob('test_*.py')):
        module=importlib.import_module('tests.legacy_pro.'+path.stem)
        for name,fn in inspect.getmembers(module,inspect.isfunction):
            if name.startswith('test_') and fn.__module__==module.__name__:
                suite.addTest(unittest.FunctionTestCase(fn,description=path.stem+'.'+name))
    start=time.time()
    with output_path('logs','tests.txt').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),seconds=time.time()-start,success=result.wasSuccessful())
    write_json(output_path('results','tests.json'),report)
    print(output_path('logs','tests.txt').read_text())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
