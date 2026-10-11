import copy
import json
from pathlib import Path
import tempfile
import unittest
from BrainPromptPro.scripts.adni_1to7 import evaluate_frozen as evaluator


class SelectionLockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'A0'/'seed42';self.lock=self.root/'lock.json'
        self.result=dict(experiment='A0',seed=42,checkpoint_sha256='checkpoint',dataset_hash='dataset',split_hash='split')
        self.doc=dict(development_complete=True,test_selection_policy='NO_TEST_BASED_SELECTION',evaluator_sha256=evaluator.sha(evaluator.__file__),protocol_hashes={'sample':'split'},evaluations=[dict(self.result,run=str(self.run),protocol='sample')])
    def write(self,doc):
        self.lock.write_text(json.dumps(doc));return evaluator.sha(self.lock)
    def test_exact_lock_is_accepted(self):
        digest=self.write(self.doc);evaluator.check_lock(self.lock,self.run,self.result,'sample',digest)
    def test_absent_or_mutated_digest_is_rejected(self):
        digest=self.write(self.doc)
        with self.assertRaises(ValueError):evaluator.check_lock(self.lock,self.run,self.result,'sample',None)
        self.lock.write_text(self.lock.read_text()+' ')
        with self.assertRaises(ValueError):evaluator.check_lock(self.lock,self.run,self.result,'sample',digest)
    def test_checkpoint_protocol_and_evaluator_are_bound(self):
        for field in ['checkpoint_sha256','split_hash','dataset_hash']:
            d=copy.deepcopy(self.doc);d['evaluations'][0][field]='changed';digest=self.write(d)
            with self.assertRaises(ValueError):evaluator.check_lock(self.lock,self.run,self.result,'sample',digest)
        d=copy.deepcopy(self.doc);d['evaluator_sha256']='changed';digest=self.write(d)
        with self.assertRaises(ValueError):evaluator.check_lock(self.lock,self.run,self.result,'sample',digest)
    def test_unfrozen_missing_and_duplicate_entries_are_rejected(self):
        for mode in ['unfrozen','missing','duplicate']:
            d=copy.deepcopy(self.doc)
            if mode=='unfrozen':d['development_complete']=False
            elif mode=='missing':d['evaluations']=[]
            else:d['evaluations']*=2
            digest=self.write(d)
            with self.assertRaises(ValueError):evaluator.check_lock(self.lock,self.run,self.result,'sample',digest)


if __name__=='__main__':unittest.main()
