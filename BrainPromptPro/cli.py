"""Independent training entrypoint, deterministic ablations and grouped CV."""
import argparse
import copy
import json
import os
import random
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data.data import LoadData
from .data.brainnet_pro import FoldDataset
from .data.metadata import MetadataEncoder, record_id
from .data.labels import require_label_policy
from .data.splits import make_splits, make_holdout, validate_split
from .metrics import classification_metrics, subject_metrics
from .nets.load_net import gnn_model
from .runtime import ROOT, output_path, write_json
from .train_TUs_graph_classification_llm import train_epoch, evaluate, save_interpretability

ABLATIONS = {'baseline': (0,0,0), 'adaptive': (1,0,0), 'disease': (0,1,0), 'multi': (0,0,1),
             'adaptive_disease': (1,1,0), 'adaptive_multi': (1,0,1), 'disease_multi': (0,1,1), 'all': (1,1,1)}
FLAGS = ('use_adaptive_brain_graph', 'use_disease_aware_readout', 'use_multirelation_population_graph')


def seed_everything(seed, deterministic):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def run_fold(config, dataset, split, fold, run_id, args, checkpoint=None):
    params, net = config['params'], config['net_params']
    seed_everything(params['seed'], params['deterministic'])
    device = torch.device(params['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
    validate_split(split, dataset.records)
    train_indices = split['train'][:args.max_train_samples] if args.max_train_samples else split['train']
    if checkpoint:
        train_indices = checkpoint['training_indices_used']
    encoder = MetadataEncoder.from_state_dict(checkpoint['metadata_encoder']) if checkpoint else MetadataEncoder().fit(
        [dataset.records[i] for i in train_indices], config['data']['site_column'], config['data']['missing_site'])
    if checkpoint:
        require_label_policy(checkpoint['provenance'].get('label_policy'), 'checkpoint provenance')
    if checkpoint and checkpoint['provenance']['sha256'] != dataset.provenance['sha256']:
        raise ValueError('Checkpoint data/prompt provenance does not match current files')
    if config['model'] == 'OriginalBrainPromptG':
        from .nets.original_adapter import OriginalBrainPromptAdapter
        model = OriginalBrainPromptAdapter(config,dataset.roi_prompt,dataset.disease_prompt).to(device)
    else:
        model = gnn_model(config['model'], net, dataset.roi_prompt, dataset.disease_prompt,
                          dataset.subject_prompt.shape[-1], len(encoder.site_mapping) + 1).to(device)
    loaders = {}
    generator = torch.Generator().manual_seed(params['seed'])
    for name, indices in split.items():
        selected = train_indices if name == 'train' else indices
        loaders[name] = DataLoader(FoldDataset(dataset, selected, encoder), batch_size=params['batch_size'],
                                   shuffle=name == 'train', num_workers=params['num_workers'], generator=generator)
    prefix = f'{run_id}/fold_{fold}'
    checkpoint_path = output_path('checkpoints', prefix, 'best.pt')
    history, best_loss, best_epoch = [], float('inf'), -1
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint['best_epoch']
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=params['lr_reduce_factor'], patience=params['lr_schedule_patience'])
        for epoch in range(params['epochs']):
            train = train_epoch(model, loaders['train'], optimizer, device, config['loss'])
            val, _ = evaluate(model, loaders['val'], device, config['loss'])
            entry = dict(epoch=epoch, train=train, validation=val, lr=optimizer.param_groups[0]['lr'])
            history.append(entry)
            print(json.dumps(dict(fold=fold, **entry), allow_nan=False), flush=True)
            with output_path('logs', prefix, 'history.jsonl').open('a') as f:
                f.write(json.dumps(entry, allow_nan=False) + '\n')
            if val['loss'] < best_loss:
                best_loss, best_epoch = val['loss'], epoch
                torch.save(dict(model=model.state_dict(), config=config, metadata_encoder=encoder.state_dict(),
                                split=split, training_indices_used=train_indices, fold=fold, best_epoch=best_epoch,
                                best_validation_loss=best_loss, provenance=dataset.provenance,
                                optimizer=optimizer.state_dict()), checkpoint_path)
            scheduler.step(val['loss'])
            patience = params.get('early_stopping_patience')
            if patience is not None and epoch-best_epoch >= patience:
                break
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                break
        best = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(best['model'])
    # No test evaluation during fitting or model selection.
    test, payload = evaluate(model, loaders['test'], device, config['loss'], capture=args.save_interpretability)
    test_subjects = subject_metrics(payload['subject_ids'],payload['true_label'].numpy(),
                                   payload['probability'].numpy(),dataset.num_classes)
    by_source = {}
    for source in sorted(set(payload['source_batches'])):
        selected = np.array([s==source for s in payload['source_batches']])
        by_source[source] = classification_metrics(payload['true_label'].numpy()[selected],
            payload['probability'].numpy()[selected],dataset.num_classes)
    # Always keep final predictions so reported metrics can be independently recomputed.
    import csv
    with output_path('results',prefix,'predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['subject_id','source_batch','true_label','predicted_label']+['p_'+c for c in dataset.class_names])
        for i,identity in enumerate(payload['subject_ids']):
            writer.writerow([identity,payload['source_batches'][i],int(payload['true_label'][i]),
                             int(payload['predicted_label'][i])]+payload['probability'][i].tolist())
    result = dict(label_policy=dataset.provenance['label_policy'],fold=fold, best_epoch=best_epoch, test=test, history=history,
                  class_names=dataset.class_names, test_subjects=test_subjects, test_by_source=by_source,
                  epochs_completed=len(history) if not checkpoint else None,
                  metadata_encoder=encoder.state_dict(), training_indices_used=train_indices,
                  split_ids={k:[record_id(dataset.records[i]) for i in v] for k,v in split.items()},
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  mode='mini_training' if len(train_indices)<len(split['train']) else 'experiment')
    write_json(output_path('results', prefix, 'result.json'), result)
    if args.save_interpretability:
        relations = net['population_relations'] if net['use_multirelation_population_graph'] else []
        save_interpretability(payload, output_path('interpretability', prefix, 'test.pt'),
                              dataset.class_names, dataset.roi_names, relations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/adni_AAL116/BrainPromptPro_adni_AAL116.json'))
    parser.add_argument('--split', type=int)
    parser.add_argument('--ablation', choices=ABLATIONS)
    parser.add_argument('--official_baseline', action='store_true', help='Use unchanged original BrainPromptG with the same safe splits')
    parser.add_argument('--device'); parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int); parser.add_argument('--seed', type=int)
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--run_id')
    parser.add_argument('--save_interpretability', action='store_true')
    parser.add_argument('--checkpoint', help='Evaluate this best.pt, with its saved configuration and split')
    parser.add_argument('--audit_only', action='store_true')
    for name in ('lambda_disease','lambda_brain_sparse','lambda_pop_sparse','lambda_relation_div'):
        parser.add_argument('--'+name, type=float)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    config = copy.deepcopy(checkpoint['config']) if checkpoint else json.loads(Path(args.config).read_text())
    if checkpoint and (args.ablation or args.official_baseline or args.max_train_samples or args.epochs or args.seed is not None):
        parser.error('Checkpoint evaluation uses saved architecture and preprocessing; training overrides are invalid')
    for name in ('device','epochs','batch_size','seed'):
        if getattr(args, name) is not None:
            config['params'][name] = getattr(args, name)
    for name in config['loss']:
        value = getattr(args, name)
        if value is not None:
            config['loss'][name] = value
        if config['loss'][name] < 0:
            parser.error('Loss weights must be nonnegative')
    if args.ablation:
        config['net_params'].update({f:bool(v) for f,v in zip(FLAGS,ABLATIONS[args.ablation])})
    if args.official_baseline:
        if args.ablation:
            parser.error('official_baseline and ablation are separate comparisons')
        config['model']='OriginalBrainPromptG'
        config['net_params'].update({f:False for f in FLAGS})
    if config['params']['epochs'] < 1 or config['params']['batch_size'] < 1:
        parser.error('epochs and batch_size must be positive')
    if args.max_train_samples is not None and args.max_train_samples < 1:
        parser.error('max_train_samples must be positive')
    torch.set_num_threads(config['params']['threads'])
    run_id = args.run_id or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ('.','..'):
        parser.error('run_id must be a simple filename')
    # Fail on incomplete full-cohort inputs before creating an experiment run.
    try:
        dataset = LoadData(config['data'])
    except ValueError as error:
        parser.error(str(error))
    run_path = output_path('results',run_id)
    run_path.mkdir(exist_ok=False)
    write_json(run_path/'config.json',config)
    write_json(run_path/'provenance.json',dataset.provenance)
    holdout = config['data'].get('split_mode') == 'grouped_holdout'
    folds = [make_holdout(dataset.records,dataset.labels.numpy(),config['params']['seed'])] if holdout else make_splits(
        dataset.records, dataset.labels.numpy(), config['params']['seed'],
        config['params']['n_splits'], config['data'].get('index_dir'))
    write_json(run_path/'splits.json',folds)
    if args.audit_only:
        print(json.dumps(dict(samples=len(dataset),subjects=dataset.provenance['subjects'],folds=len(folds))))
        return
    if checkpoint:
        numbers = [checkpoint['fold']]
        if args.split is not None and args.split != numbers[0]:
            parser.error('Requested split differs from checkpoint')
    else:
        numbers = range(len(folds)) if args.split is None else [args.split]
    if any(i < 0 or i >= len(folds) for i in numbers):
        parser.error('split is outside the configured fold range')
    results = [run_fold(config,dataset,checkpoint['split'] if checkpoint else folds[i],i,run_id,args,checkpoint)
               for i in numbers]
    summary = dict(label_policy=dataset.provenance['label_policy'],cohort_name=dataset.provenance['cohort_name'],
                   scope=dataset.provenance['scope'], samples=len(dataset),
                   subjects=dataset.provenance['subjects'],
                   folds=[r['fold'] for r in results],
                   accuracy_mean=float(np.mean([r['test']['accuracy'] for r in results])),
                   accuracy_std=float(np.std([r['test']['accuracy'] for r in results])),
                   evaluation_protocol=config['data'].get('split_mode','grouped'),
                   complete_cv=not holdout and len(results)==len(folds),
                   full_adni_cv=(config['data'].get('scope')=='full_adni'
                                 and len(results)==len(folds)
                                 and all(r['mode']=='experiment' for r in results)))
    write_json(run_path/'summary.json',summary)
    print(json.dumps(summary),flush=True)
