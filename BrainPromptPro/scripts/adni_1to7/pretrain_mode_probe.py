"""TRAIN-only diagnostic of dropout/BatchNorm reconstruction mismatch."""
import argparse
import json
from pathlib import Path
import torch
from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
from BrainPromptPro.layers.masked_connectome import symmetric_fc_mask
from BrainPromptPro.nets.connectome_p2p3p5 import ConnectomeBrainPrompt
from BrainPromptPro.scripts.adni_1to7.reconstruction_probe import metrics
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads((args.run / 'result.json').read_text())
    checkpoint = args.run / 'best_training.pt'
    if result['smoke_epochs'] or result['stage'] != 'pretrain' or sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Completed formal pretraining checkpoint required')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg, spec = ck['config'], ck['spec']
    dataset = PreparedCohortDataset(cfg['data'])
    split_path = Path(cfg['study']['sample_split'])
    if sha(split_path) != ck['split_hash']:
        raise ValueError('Split mismatch')
    split = json.loads(split_path.read_text())
    train = torch.tensor(split['train'][:128])
    torch.set_num_threads(2)
    modes = {}
    upper = torch.ones(116, 116, dtype=torch.bool).triu(1)
    for mode in ['eval', 'train', 'batch_stats_no_dropout', 'running_stats_with_dropout']:
        torch.manual_seed(4721)
        model = ConnectomeBrainPrompt(cfg, dataset.roi_prompt, dataset.disease_prompt, spec['modules'])
        model.load_state_dict(ck['model'])
        model.train(mode in ['train', 'batch_stats_no_dropout'])
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p=spec.get('pretrain_dropout',module.p)
                module.train(mode in ['train', 'running_stats_with_dropout'])
        g = torch.Generator().manual_seed(91201)
        predictions, targets, variation = [], [], []
        with torch.no_grad():
            for index in train.split(20):
                fc = dataset.fc[index]
                masked, visible = symmetric_fc_mask(fc, spec['modules']['mask_ratio'], g)
                roi, _ = model.encode_roi(masked, visible)
                prediction = model.decoder(roi)
                take = (~visible) & upper
                predictions.append(prediction[take]);targets.append(fc[take])
                variation.append(((roi-roi.mean(1,keepdim=True)).square().mean((1,2))/roi.square().mean((1,2)).clamp_min(1e-12)))
        modes[mode] = metrics(torch.cat(predictions), torch.cat(targets))
        modes[mode]['relative_roi_variation'] = float(torch.cat(variation).mean())
    report = dict(experiment=ck['experiment'], evaluation='TRAIN diagnostic only', modes=modes,
                  checkpoint_sha256=sha(checkpoint), source_sha256=sha(Path(__file__)),
                  checkpoint_modified=False, test_accessed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k:round(v['masked_mse'],6) for k,v in modes.items()}))


if __name__ == '__main__':
    main()
