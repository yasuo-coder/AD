#!/usr/bin/env python3
"""Inventory all seven ADNI batches; creates no training data or exclusions."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cohort_inventory', ROOT / 'data/cohort.py')
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


def main():
    config = json.loads((ROOT / 'configs/adni_AAL116/BrainPromptPro_adni_AAL116.json').read_text())['data']
    report, rows = cohort.inspect_cohort(config['source_root'], config['index_dir'])
    output = ROOT / 'outputs/results/full_adni_audit'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'scan_inventory.csv').open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(report=str(output / 'inventory.json'),
        scans=report['inventory_scan_rows'], subjects=report['unique_subjects'],
        status=report['status'])))


if __name__ == '__main__':
    main()
