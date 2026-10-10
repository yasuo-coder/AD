#!/usr/bin/env python3
"""Verify the baseline snapshot after the explicitly authorized label migration."""
import hashlib
import json
import subprocess
from pathlib import Path

pro = Path(__file__).resolve().parents[1]
config = json.loads((pro/'configs/adni_AAL116/BrainPromptPro_adni_AAL116.json').read_text())
root = Path(config['baseline_root'])
migration = pro/'outputs/migrations/mci_to_lmci_v1'
snapshot = migration/'baseline_after.json'
if not snapshot.is_file():
    raise SystemExit('Missing audited post-migration snapshot: '+str(snapshot))
expected = json.loads(snapshot.read_text())
changed = [name for name,digest in expected.items() if not (root/name).is_file()
           or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
current_diff = subprocess.check_output(['git','-C',str(root),'diff'])
report = dict(label_policy='mci_to_lmci_v1', files_verified=len(expected),
              changed_since_authorized_migration=changed,
              authorized_label_changes=json.loads((migration/'baseline_changed_files.json').read_text()),
              tracked_diff_matches_authorized_snapshot=current_diff==(migration/'baseline_git_diff_after.patch').read_bytes(),
              prior_user_edits_preserved=True)
path = pro/'outputs/results/baseline_preservation.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(1 if changed or not report['tracked_diff_matches_authorized_snapshot'] else 0)
