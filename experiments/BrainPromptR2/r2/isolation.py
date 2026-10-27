"""Allowlisted filesystem, private PID/network namespaces, FIT-only data mount."""
import os
from pathlib import Path
from .common import ROOT,OLD
PYTHON=Path('/home/panyao/miniconda3/envs/brainprompt/bin/python')
def command(task,output,gpu='4',extra=None,code=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cmd=['/usr/bin/bwrap','--die-with-parent','--new-session','--unshare-pid','--unshare-net']
    for system in ['/usr','/lib','/lib64','/bin','/etc','/sys']:
        if Path(system).exists():cmd+=['--ro-bind',system,system]
    cmd+=['--proc','/proc','--dev-bind','/dev','/dev','--tmpfs','/tmp','--tmpfs','/run',
      '--ro-bind',str(PYTHON.parents[2]),str(PYTHON.parents[2]),
      '--ro-bind',str(ROOT/'r2'),'/work/code/r2','--ro-bind',str(ROOT/'configs'),'/work/code/configs',
      '--ro-bind',str(ROOT/'legacy'),'/work/code/legacy',
      '--ro-bind',str(ROOT/'public/source_manifest.json'),'/work/code/source_manifest.json',
      '--ro-bind',str(ROOT/'prepared'),'/work/data','--ro-bind',str(Path(task).resolve()),'/work/task.json',
      '--bind',str(output.resolve()),'/work/out','--chdir','/work/code',
      '--clearenv','--setenv','PATH',str(PYTHON.parent)+':/usr/bin:/bin','--setenv','PYTHONPATH','/work/code',
      '--setenv','PYTHONDONTWRITEBYTECODE','1','--setenv','CUDA_VISIBLE_DEVICES',str(gpu),
      '--setenv','CUBLAS_WORKSPACE_CONFIG',':4096:8','--setenv','OMP_NUM_THREADS','2',
      '--setenv','OPENBLAS_NUM_THREADS','2','--setenv','MKL_NUM_THREADS','2','--setenv','XDG_CACHE_HOME','/tmp/cache',
      '--setenv','DGLBACKEND','pytorch','--',str(PYTHON),'-B','-u']
    if code is not None:return cmd+['-c',code]
    return cmd+['-m','r2.train']+(extra or [])
