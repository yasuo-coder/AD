import json
from pathlib import Path
import tempfile
import unittest

from BrainPromptPro.scripts.adni_1to7.summarize_validation import queue_activity


class QueueSummaryTests(unittest.TestCase):
    def write(self, root, name, state):
        p = root / name
        p.mkdir()
        (p / 'state.private.json').write_text(json.dumps(state))

    def test_mixed_controllers_and_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 'train', dict(queue_pid=10, running=[dict(experiment='A0', seed=42)],
                pending=[dict(experiment='A1', seed=3407, split='1')]))
            self.write(root, 'evaluation', dict(pid=11, phase='VERIFYING_DEVELOPMENT'))
            self.write(root, 'failed_evaluation', dict(pid=12, phase='FAILED'))
            self.assertEqual(queue_activity(root, 'sample'), ({'A0': {42}}, {}))
            self.assertEqual(queue_activity(root, 'subject'), ({}, {'A1': {3407}}))

    def test_malformed_training_queue_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 'train', dict(queue_pid=10, phase='FAILED', pending=[]))
            with self.assertRaises(ValueError):
                queue_activity(root, 'sample')

    def test_unknown_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 'unknown', dict(status='unknown'))
            with self.assertRaises(ValueError):
                queue_activity(root, 'sample')


if __name__ == '__main__':
    unittest.main()
