"""Validation-only masked-FC probe against TRAIN-fitted constant baselines."""
import argparse
import json
from pathlib import Path

import torch

from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
from BrainPromptPro.layers.masked_connectome import symmetric_fc_mask
from BrainPromptPro.nets.connectome_p2p3p5 import ConnectomeBrainPrompt
from BrainPromptPro.research_runner import seed_all
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha, write_json


def metrics(prediction, target):
    prediction = prediction.double()
    target = target.double()
    error = (prediction - target).square()
    pos, neg = target > 0, target < 0
    return dict(
        masked_edges=len(target),
        masked_mse=float(error.mean()),
        positive_mse=float(error[pos].mean()) if pos.any() else None,
        negative_mse=float(error[neg].mean()) if neg.any() else None,
        correlation=float(torch.corrcoef(torch.stack((prediction, target)))[0, 1])
        if prediction.std() > 1e-12 and target.std() > 1e-12 else None,
        prediction_mean=float(prediction.mean()),
        prediction_std=float(prediction.std()),
        target_mean=float(target.mean()),
        target_std=float(target.std()),
        saturated_fraction=float((prediction.abs() > .99).double().mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads((args.run / 'result.json').read_text())
    if result['stage'] != 'pretrain' or result['smoke_epochs']:
        raise ValueError('A completed formal pretraining run is required')
    checkpoint = args.run / 'best_training.pt'
    if sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg, spec = ck['config'], ck['spec']
    dataset = PreparedCohortDataset(cfg['data'])
    split_path = Path(cfg['study']['sample_split'])
    if sha(split_path) != ck['split_hash']:
        raise ValueError('Only the registered sample-level split is supported')
    split = json.loads(split_path.read_text())
    # All fitted comparators use TRAIN FC; no test loader or test metrics.
    mean_edge = dataset.fc[split['train']].mean(0)
    upper = torch.ones(116, 116, dtype=torch.bool).triu(1)
    mean_global = float(mean_edge[upper].mean())
    torch.set_num_threads(2)
    device = torch.device(args.device)
    seed_all(ck['seed'])
    model = ConnectomeBrainPrompt(cfg, dataset.roi_prompt, dataset.disease_prompt, spec['modules']).to(device).eval()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    collected = {}
    # Fixed probe masks are shared by all decoders/runs for each ratio.
    for variant in ['initial', 'best_training', 'last']:
        if variant == 'initial':
            model.load_state_dict(initial)
        else:
            loaded = ck if variant == 'best_training' else torch.load(args.run / 'last.pt', map_location='cpu', weights_only=True)
            if loaded['signature'] != ck['signature']:
                raise ValueError('Last/best run signatures differ')
            model.load_state_dict(loaded['model'])
        predictions, targets, means, classes = [], [], [], []
        with torch.no_grad():
            for mask_seed in [91201, 91202, 91203]:
                generator = torch.Generator(device=device).manual_seed(mask_seed)
                for index in torch.tensor(split['val']).split(20):
                    fc = dataset.fc[index].to(device)
                    masked, visible = symmetric_fc_mask(fc, spec['modules']['mask_ratio'], generator)
                    roi, _ = model.encode_roi(masked, visible)
                    prediction = model.decoder(roi)
                    hidden = (~visible) & upper.to(device)
                    predictions.append(prediction[hidden].cpu())
                    targets.append(fc[hidden].cpu())
                    means.append(mean_edge.to(device).expand_as(fc)[hidden].cpu())
                    classes.append(dataset.labels[index].to(device)[:, None, None].expand_as(fc)[hidden].cpu())
        prediction, target, edge_mean, label = map(torch.cat, [predictions, targets, means, classes])
        collected[variant] = metrics(prediction, target)
        collected[variant]['by_class'] = {
            name: metrics(prediction[label == i], target[label == i])
            for i, name in enumerate(cfg['data']['class_names']) if (label == i).any()
        }
        if variant == 'best_training':
            collected['train_edge_mean'] = metrics(edge_mean, target)
            collected['train_global_mean'] = metrics(torch.full_like(target, mean_global), target)
            collected['zero'] = metrics(torch.zeros_like(target), target)
    report = dict(
        experiment=ck['experiment'], seed=ck['seed'], split='validation_only',
        checkpoint_sha256=result['checkpoint_sha256'], dataset_hash=ck['dataset_hash'], split_hash=ck['split_hash'],
        best_training_epoch=ck['epoch'] + 1, mask_ratio=spec['modules']['mask_ratio'],
        fixed_mask_seeds=[91201, 91202, 91203], result=collected, test_accessed=False,
        probe_source_sha256=sha(Path(__file__)),
        beats_train_edge_mean=collected['best_training']['masked_mse'] < collected['train_edge_mean']['masked_mse'],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ['experiment', 'beats_train_edge_mean', 'test_accessed']}))


if __name__ == '__main__':
    main()
