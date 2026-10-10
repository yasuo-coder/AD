"""Three fixed architectures on merged ADNI_1--4; identical official outer protocol."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from .data.data import LoadData
from .data.brainnet_pro import FoldDataset, sha256
from .data.metadata import MetadataEncoder, record_id
from .losses import compute_losses
from .metrics import classification_metrics
from .runtime import ROOT, output_path
from .train_TUs_graph_classification_llm import model_inputs

UPSTREAM_COMMIT = 'c85304a023636a2701d465e4cb0acb13b637127e'
FLAGS = ('use_adaptive_brain_graph', 'use_disease_aware_readout', 'use_multirelation_population_graph')


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


from .official_cv import make_official_splits, audit_splits, stop_reason, epoch_pass
from .official_cv import source_hashes as official_source_hashes

MODELS = ('BrainPrompt', 'BrainPromptPro_old', 'BrainPromptPro_new')


def source_hashes():
    paths = [ROOT/'merged_cv.py', ROOT/'scripts/prepare_adni1234.py']
    return dict(official_source_hashes(), **{str(p):sha256(p) for p in paths})


def method_config(base, name):
    config=copy.deepcopy(base)
    if name=='BrainPrompt':
        config['model']='OriginalBrainPromptG'
        config['net_params'].update({f:False for f in FLAGS})
    elif name=='BrainPromptPro_old':
        config['net_params']['stability_profile']='legacy'
    elif name=='BrainPromptPro_new':
        config['net_params'].update(stability_profile='stable_v1',hidden_dim=64,L=2,
            brain_edge_hidden_dim=32,disease_attention_dim=64,relation_hidden_dim=32,batch_norm=False)
    else:
        raise ValueError(name)
    return config


def prepare(run_id):
    import shutil, platform, dgl, sklearn, scipy
    config=json.loads((ROOT/'configs/adni1234/BrainPromptPro_adni1234_download_merged.json').read_text())
    config['data']['split_mode']='official_scan_stratified_10fold'
    config['params'].update(epochs=1000,n_splits=10,early_stopping_patience=None,
        device='cuda:0',max_time=60,checkpoint_selection='last_epoch',deterministic_algorithms=False,
        label_smoothing=0.,class_weight_power=0.,gradient_clip_norm=None)
    torch.set_num_threads(config['params']['threads'])
    data=LoadData(config['data']);folds=make_official_splits(data.labels.numpy(),config['params']['seed'])
    audit=audit_splits(folds,data.records)
    root=output_path('comparisons',run_id);root.mkdir(exist_ok=False)
    dump(root/'config.json',config);dump(root/'splits.json',folds);dump(root/'split_audit.json',audit)
    dump(root/'provenance.json',data.provenance);hashes=source_hashes();dump(root/'source_sha256.json',hashes)
    for name, digest in hashes.items():
        p=Path(name);dest=root/'source_training'/p.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha256(dest)==digest
    configs={name:method_config(config,name) for name in MODELS}
    assert all(c['params']==config['params'] and c['loss']==config['loss'] and c['data']==config['data'] for c in configs.values())
    assert all(c['net_params']['dropout']==.5 for c in configs.values())
    for name,c in configs.items():dump(root/(name+'_config.json'),c)
    dump(root/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        dgl=dgl.__version__,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__))
    dump(root/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),upstream_commit=UPSTREAM_COMMIT,
        models=list(MODELS),input='Merged ADNI_1+2+3+4: 798 preprocessed scans; source CSV Group; MCI -> LMCI',
        selection='Fixed architecture for all ten folds; actual final epoch; no best restoration or candidate selection',
        split='Scan-level StratifiedKFold(10,shuffle=True,seed=41), stratified validation test_size=0.111',
        training_order='Independent torch.Generator seed=41 for each training loader, identical per-epoch scan order across architectures',
        loss_aggregation='Unweighted mean of batch losses, matching official trainer',
        common_loss=config['loss'],common_training=config['params'],common_dropout=.5,
        architecture='A/B hidden=116,L=4; C stable_v1 hidden=64,L=2 and LayerNorm; architecture differences only',
        test_monitoring='Twice per epoch as upstream; never used for scheduling, stopping, tuning or selection',
        scheduler='Adam; validation total loss ReduceLROnPlateau factor=.5 patience=25; stop lr<1e-6',
        differences_from_unmodified_upstream=[
            'Audited merged subset, not full paper cohort; download Group labels with MCI merged into LMCI',
            'Seeded split RNG and independent seeded training-shuffle RNG shared across methods',
            'Original BrainPromptG model unchanged, common adapter, last state_dict checkpoint',
            'Honor baseline.sh edge_ratio=.2; upstream second dataset load drops this argument',
            'Correct macro metrics and probability-based OvR AUC replace upstream multiclass placeholders',
            'Pro brain/population sparsity losses retain common 1e-4 weights; absent learned modules contribute zero in A'],
        note_on_new='Only stable_v1 architecture carried over. No class weighting, smoothing, clip, reduced auxiliary coefficient, per-fold fallback, or new-only dropout.',
        caution='Longitudinal subject overlap is measured; this scan-level protocol does not estimate unseen-subject generalization',
        batch_size=20,seed=41,epochs_cap=1000,per_fold_time_cap_hours=6,ddof=0))
    with (root/'cohort.csv').open('w',newline='') as stream:
        keys=['scan_index','subject_id','source_batch','true_label','diagnosis','fmri_image_id']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for i,row in enumerate(data.records):
            writer.writerow(dict(scan_index=i,subject_id=record_id(row),source_batch=row['source_batch'],
                true_label=row['label'],diagnosis=row['diagnosis'],fmri_image_id=row['fmri_image_id']))
    print(json.dumps(dict(run_root=str(root),samples=len(data),subjects=data.provenance['subjects'],
        class_counts=np.bincount(data.labels.numpy(),minlength=5).tolist(),
        overlapping_test_scans=sum(a['test_scans_with_train_subject'] for a in audit))),flush=True)


def worker(run_id, name, fold, smoke=False):
    root = output_path('comparisons', run_id)
    if source_hashes() != json.loads((root / 'source_sha256.json').read_text()):
        raise ValueError('Source changed after experiment preparation')
    config = method_config(json.loads((root / 'config.json').read_text()), name)
    if smoke: config['params']['epochs']=3
    params = config['params']
    torch.set_num_threads(params['threads'])
    data = LoadData(config['data'])
    folds = json.loads((root / 'splits.json').read_text())
    audit_splits(folds, data.records)
    split = folds[fold]
    out = (root/'smoke' if smoke else root) / name / ('fold_' + str(fold))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'config.json', config)
    start = time.monotonic()
    import dgl
    seed = params['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dgl.seed(seed); dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    device = torch.device(params['device'])
    if not torch.cuda.is_available():
        raise RuntimeError('This full experiment requires CUDA')
    encoder = MetadataEncoder().fit([data.records[i] for i in split['train']], config['data']['site_column'], config['data']['missing_site'])
    if name == 'BrainPrompt':
        from .nets.original_adapter import OriginalBrainPromptAdapter
        model = OriginalBrainPromptAdapter(config, data.roi_prompt, data.disease_prompt).to(device)
    else:
        from .nets.load_net import gnn_model
        model = gnn_model(config['model'], config['net_params'], data.roi_prompt, data.disease_prompt,
                          data.subject_prompt.shape[-1], len(encoder.site_mapping) + 1).to(device)
    shuffle_generator=torch.Generator().manual_seed(seed)
    loaders = {k: DataLoader(FoldDataset(data, v, encoder), batch_size=params['batch_size'],
                            shuffle=k == 'train', num_workers=0, drop_last=False,
                            generator=shuffle_generator if k=='train' else None) for k, v in split.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=params['lr_reduce_factor'], patience=params['lr_schedule_patience'])
    history = []
    for epoch in range(params['epochs']):
        epoch_start = time.monotonic()
        lr = optimizer.param_groups[0]['lr']
        train, _ = epoch_pass(model, loaders['train'], device, config['loss'], optimizer)
        val, _ = epoch_pass(model, loaders['val'], device, config['loss'])
        test, _ = epoch_pass(model, loaders['test'], device, config['loss'])
        test_again, _ = epoch_pass(model, loaders['test'], device, config['loss'])
        # CUDA graph reductions can vary at floating-point precision. Upstream
        # does not require bitwise equality and this must not stop training.
        repeated_test_delta = abs(test['loss'] - test_again['loss'])
        scheduler.step(val['loss'])
        elapsed = time.monotonic() - start
        reason = stop_reason(epoch, optimizer.param_groups[0]['lr'], elapsed, params)
        entry = dict(epoch=epoch, epoch_number=epoch+1, train=train, validation=val, test=test,
                     lr=lr, next_lr=optimizer.param_groups[0]['lr'], elapsed_seconds=elapsed,
                     epoch_seconds=time.monotonic()-epoch_start, stop_reason=reason,
                     repeated_test_loss_abs_delta=repeated_test_delta)
        history.append(entry)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
        checkpoint = dict(model=model.state_dict(), config=config, metadata_encoder=encoder.state_dict(),
                          split=split, training_indices_used=split['train'], fold=fold, last_epoch=epoch,
                          epochs_completed=epoch+1, checkpoint_selection='last_epoch', provenance=data.provenance,
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), stop_reason=reason,
                          split_sha256=sha256(root / 'splits.json'), upstream_commit=UPSTREAM_COMMIT)
        torch.save(checkpoint, out / 'last.tmp.pt')
        (out / 'last.tmp.pt').replace(out / 'last.pt')
        dump(out / 'status.json', dict(model=name, fold=fold, **entry))
        if epoch == 0 or (epoch+1) % 20 == 0 or reason:
            print(json.dumps(dict(model=name, fold=fold, epoch=epoch+1, lr=lr,
                                  train_loss=train['loss'], val_loss=val['loss'], test_accuracy=test['accuracy'],
                                  elapsed_seconds=elapsed, stop_reason=reason)), flush=True)
        if reason:
            break
    final_train, _ = epoch_pass(model, loaders['train'], device, config['loss'])
    # Load the actual on-disk last checkpoint, never a validation-best model.
    last = torch.load(out / 'last.pt', map_location=device, weights_only=True)
    if last['last_epoch'] != history[-1]['epoch']:
        raise RuntimeError('Final checkpoint epoch mismatch')
    model.load_state_dict(last['model'])
    final_test, payload = epoch_pass(model, loaders['test'], device, config['loss'], capture=True)
    expected = [record_id(data.records[i]) for i in split['test']]
    if payload['ids'] != expected:
        raise ValueError('Test IDs/order mismatch')
    y, prob = payload['truth'].numpy(), payload['probability'].numpy()
    metrics = classification_metrics(y, prob, data.num_classes)
    metrics['loss'] = final_test['loss']
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scan_index','subject_id','source_batch','true_label','predicted_label'] + ['p_' + c for c in data.class_names])
        for j, i in enumerate(split['test']):
            writer.writerow([i, expected[j], data.records[i]['source_batch'], int(y[j]), int(prob[j].argmax())] + prob[j].tolist())
    result = dict(model=name, fold=fold, class_names=data.class_names, test=metrics, final_train=final_train,
                  epochs_completed=len(history), last_epoch=history[-1]['epoch'], checkpoint_selection='last_epoch',
                  checkpoint=str(out/'last.pt'), checkpoint_sha256=sha256(out/'last.pt'),
                  stop_reason=reason, final_lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-start,
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name(0),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  metadata_encoder=encoder.state_dict(), split_sha256=sha256(root/'splits.json'),
                  bundle_sha256=config['data']['bundle_sha256'])
    result['smoke']=smoke
    dump(out / 'result.json', result)
    print(json.dumps(dict(event='complete', model=name, fold=fold, test=metrics, epochs=len(history))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','worker'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--fold', type=int, choices=range(10))
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('.','..'):
        parser.error('Invalid run ID')
    if args.command == 'prepare':
        prepare(args.run_id)
    else:
        if args.model is None or args.fold is None:
            parser.error('worker requires --model and --fold')
        worker(args.run_id, args.model, args.fold, args.smoke)


if __name__ == '__main__':
    main()
