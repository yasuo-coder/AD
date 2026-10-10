"""Import and private-exception checks use synthetic inputs only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from BrainPromptPro.data.private_audit import expected_ids
from BrainPromptPro.data.labels import CLASS_NAMES,normalize_diagnosis

class IntegrationTests(unittest.TestCase):
    def test_private_audit_missing_fails_closed(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):expected_ids('missing_original_images')

    def test_private_audit_validates_adjudicated_sets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'private.json'
            with patch.dict(os.environ,{'BRAINPROMPT_PRIVATE_AUDIT_JSON':str(p)}):
                for values in [[],None,{},[''],['SYNTHETIC','SYNTHETIC']]:
                    p.write_text(json.dumps({'missing_original_images':values}))
                    with patch('BrainPromptPro.data.private_audit.AUDIT_SHA256', hashlib.sha256(p.read_bytes()).hexdigest()):
                        with self.assertRaises(ValueError):expected_ids('missing_original_images')
                p.write_text(json.dumps({'missing_original_images':['SYNTHETIC_A','SYNTHETIC_B']}))
                with patch('BrainPromptPro.data.private_audit.AUDIT_SHA256', hashlib.sha256(p.read_bytes()).hexdigest()):
                    self.assertEqual(expected_ids('missing_original_images'),{'SYNTHETIC_A','SYNTHETIC_B'})
                    p.write_text(p.read_text()+' ')
                    with self.assertRaises(ValueError):expected_ids('missing_original_images')

    def test_label_and_layout_contracts_preserved(self):
        self.assertEqual(list(CLASS_NAMES),['CN','AD','SMC','EMCI','LMCI'])
        self.assertEqual(normalize_diagnosis('MCI'),'LMCI')
        self.assertEqual(normalize_diagnosis('EMCI'),'EMCI')
        root=Path(__file__).resolve().parents[1]
        self.assertFalse((root/'BrainPromptPro/BrainPromptPro').exists())
        for p in (root/'BrainPromptPro').rglob('*.py'):
            if 'outputs' not in p.parts:ast.parse(p.read_text(),filename=str(p))

if __name__=='__main__':unittest.main()
