"""CPU-only synthetic regression subset for the imported public repository."""
import importlib
import inspect
import unittest
import torch

def main():
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_public_integration')
    names=['test_shapes','test_gradients','test_adaptive','test_readout','test_graph_symmetry','test_no_label_leakage','test_stable_profile','test_holdout_metrics']
    for name in names:
        module=importlib.import_module('tests.legacy_pro.'+name)
        for key,fn in inspect.getmembers(module,inspect.isfunction):
            if key.startswith('test_') and fn.__module__==module.__name__:
                suite.addTest(unittest.FunctionTestCase(fn,description=name+'.'+key))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

if __name__=='__main__':main()
