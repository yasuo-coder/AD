"""Wait for one declared training queue, then verify and summarize validation.

This observer never launches training, changes a checkpoint, or evaluates TEST.
Run from a sealed code snapshot; record its PID and log like the GPU controller.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--protocol',choices=['sample','subject'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True)
    a=p.parse_args();code=Path(__file__).resolve().parents[3]
    if not a.output.resolve().is_relative_to(a.root.resolve()/'reports'):raise ValueError('Reports must stay inside the study')
    a.status.parent.mkdir(parents=True,exist_ok=True)
    lock=a.status.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.output.exists() or a.status.exists():raise FileExistsError('Use new output/status for a reviewed recovery')
    paths=[Path(__file__),Path(__file__).with_name('summarize_validation.py'),Path(__file__).with_name('analyze_ablation.py')]
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    def save(phase,**extra):
        temporary=a.status.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(dict(pid=os.getpid(),phase=phase,protocol=a.protocol,queue=str(a.queue),output=str(a.output),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=hashes,**extra),indent=2)+'\n');temporary.replace(a.status)
    try:
        while True:
            q=json.loads(a.queue.read_text())
            if q['failed']:raise RuntimeError('Dependency queue failed; inspect before recovery')
            if not q['running'] and not q['pending']:break
            try:os.kill(q['queue_pid'],0)
            except ProcessLookupError:raise RuntimeError('Dependency controller disappeared before completion')
            save('WAITING',finished=len(q['finished']),running=len(q['running']),pending=len(q['pending']))
            time.sleep(10)
        if any(hashlib.sha256(Path(x).read_bytes()).hexdigest()!=h for x,h in hashes.items()):raise RuntimeError('Observer source changed')
        save('VERIFYING')
        subprocess.run([sys.executable,'-B','-m','BrainPromptPro.scripts.adni_1to7.summarize_validation','--root',str(a.root),'--protocol',a.protocol,'--output',str(a.output)],cwd=code,check=True)
        if a.protocol=='sample':
            subprocess.run([sys.executable,'-B','-m','BrainPromptPro.scripts.adni_1to7.analyze_ablation','--summary',str(a.output/'validation_summary.public.json'),'--output',str(a.output/'ablation')],cwd=code,check=True)
        save('COMPLETE',test_accessed=False)
    except Exception as exc:
        save('FAILED',error=str(exc));raise


if __name__=='__main__':main()
