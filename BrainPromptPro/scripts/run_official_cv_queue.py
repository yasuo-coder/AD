"""Run twenty independent folds on explicitly selected GPUs; no retry/resume mutation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', required=True)
parser.add_argument('--gpus', nargs='+', default=['5','6','7'])
parser.add_argument('--jobs-per-gpu', type=int, default=2)
args = parser.parse_args()
root = Path('/home/panyao/AD/BrainPromptPro/outputs/comparisons') / args.run_id
queue = [(model, fold) for fold in range(10) for model in ['BrainPrompt','BrainPromptPro']]
slots = [(gpu, slot) for slot in range(args.jobs_per_gpu) for gpu in args.gpus]
active, completed, failed = {}, [], []
(root/'worker_logs').mkdir(exist_ok=False)
start = time.monotonic()
while queue or active:
    for slot, item in list(active.items()):
        status = item['process'].poll()
        if status is None:
            continue
        item['log'].close()
        result = dict(model=item['model'], fold=item['fold'], gpu=slot[0], returncode=status)
        (completed if status == 0 else failed).append(result)
        print(json.dumps(dict(event='completed' if status==0 else 'failed', **result)), flush=True)
        del active[slot]
    if not failed:
        for slot in slots:
            if slot in active or not queue:
                continue
            model, fold = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0], PYTHONHASHSEED='41',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       MPLCONFIGDIR=str(root/'matplotlib_cache'))
            log = (root/'worker_logs'/f'{model}_fold_{fold}.log').open('w')
            command = [sys.executable, '-m', 'BrainPromptPro.official_cv', 'worker',
                       '--run-id', args.run_id, '--model', model, '--fold', str(fold)]
            process = subprocess.Popen(command, cwd='/home/panyao/AD', env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            active[slot] = dict(process=process, log=log, model=model, fold=fold)
            print(json.dumps(dict(event='started', model=model, fold=fold, gpu=slot[0], pid=process.pid)), flush=True)
    status = dict(completed=completed, failed=failed, pending=queue,
                  active=[dict(model=i['model'], fold=i['fold'], gpu=s[0], pid=i['process'].pid)
                          for s,i in active.items()], elapsed_seconds=time.monotonic()-start)
    temporary = root/'queue_status.tmp.json'
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    temporary.replace(root/'queue_status.json')
    if failed and not active:
        break
    if active:
        time.sleep(10)
print(json.dumps(dict(event='queue_finished', completed=len(completed), failed=failed,
                      elapsed_seconds=time.monotonic()-start)), flush=True)
sys.exit(1 if failed else 0)
