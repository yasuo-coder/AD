#!/usr/bin/env python3
"""Read-only validation ablation calculations; writes evidence, never selection locks.

Four-context mean marginal is an equally weighted factorial/Banzhaf contrast,
not a Shapley-weighted contrast and not a substitute for full-minus-one. All
means/SD use the same three seeds; contexts are not independent replicates.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/panyao/AD/BrainPromptPro')
RUN = ROOT/'outputs/research/final_optimization_20261007'
STAGE = RUN/'stage2_20261008'
SEEDS = [42, 3407, 2026]
PROFILE = 'sqrt_inverse_frequency'
METRICS = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']
CATEGORY_ORDER = {'Strong': 0, 'Acceptable': 1, 'Mixed': 2, 'Weak': 3}
FAMILIES = {
    'M1_M3_M5': ['M1', 'M3_v2', 'M5'],
    'M1_M4_M5': ['M1', 'M4', 'M5'],
    'M1_M2_M3v3': ['M1', 'M2', 'M3_v3'],
    'M1_M2_M5': ['M1', 'M2', 'M5'],
}


def require(ok, text):
    if not ok:
        raise AssertionError(text)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    require(len(a) == 3 and np.isfinite(a).all(), 'Exactly three finite seed observations required')
    return dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)), per_seed={str(s): float(v) for s, v in zip(SEEDS, a)})


def category(delta):
    # Same numerical-zero convention as existing summarize_stage2.py.
    positive = sum(delta[k] > 1e-12 for k in METRICS)
    minimum = min(delta.values())
    return ('Strong' if positive == 5 else 'Acceptable' if positive >= 4 and minimum >= -.01 else
            'Weak' if positive <= 2 and minimum < -.01 else 'Mixed')


def composite(m):
    return .2*m['accuracy']+.25*m['f1_macro']+.2*m['precision_macro']+.2*m['recall_macro']+.15*m['roc_auc_macro']


def atoms(spec):
    """Use explicit mechanism versions; generic registry version is not M3 version."""
    result = []
    for module in spec.get('modules', []):
        result.append(('M3_v3' if spec.get('population_version') == 3 else 'M3_v2') if module == 'M3' else module)
    return frozenset(result)


def frozen_parameters(out, spec):
    control = spec.get('control', 'learned')
    if spec.get('population_version') != 3 or control == 'learned':
        return 0
    ck = torch.load(out/'best_validation.pt', map_location='cpu', weights_only=True)
    require(ck['spec'] == spec, 'V3 cost checkpoint/spec match')
    state = ck['model']
    count = sum(v.numel() for k, v in state.items() if k.startswith('population.scorers.') and torch.is_tensor(v))
    if control in ('fixed', 'direct'):
        count += sum(state['population.'+k].numel() for k in ('metric', 'temperature', 'fusion_logits'))
    require(control in ('fixed', 'direct', 'no_metadata'), 'Known V3 control for frozen counts')
    return int(count)


def load_verification_index():
    index = {}
    for path in (RUN/'verification/development_verification.json', STAGE/'verification/development_verification.json'):
        if not path.is_file():
            continue
        report = read(path)
        if report.get('failures') or report.get('passed') is False:
            continue
        for item in report.get('experiments', []):
            if 'checkpoint_sha256' in item:
                index[item['checkpoint_sha256']] = dict(report=str(path), report_sha256=sha(path))
    return index


def collect_model(name, owner, spec, protocol, verification):
    values = {}; pending = []
    for seed in SEEDS:
        out = owner/'experiments'/name/PROFILE/f'seed_{seed}'; path = out/'result.json'
        if not path.is_file():
            pending.append(seed); continue
        r = read(path)
        require(r['model'] == name and r['seed'] == seed and r['supporting_profile'] == PROFILE, f'{path}: identity')
        require(r['test'] == 'NOT_EVALUATED' and r['split_sha256'] == protocol['split_sha256'], f'{path}: development/split marker')
        require(read(out/'spec.json') == spec, f'{path}: frozen registry spec')
        require(sha(out/'best_validation.pt') == r['checkpoint_sha256'], f'{path}: checkpoint hash')
        require(all(np.isfinite(r['val'][k]) for k in METRICS+['loss']), f'{path}: finite validation metrics')
        total = int(r['parameter_count'])
        frozen = frozen_parameters(out, spec) if owner == STAGE else 0
        values[seed] = dict(metrics={k: float(r['val'][k]) for k in METRICS}, train_metrics={k: float(r['train'][k]) for k in METRICS},
                            composite=composite(r['val']), loss=float(r['val']['loss']),
                            parameter_count_total=total, parameter_count_trainable=total-frozen,
                            explicit_frozen_parameter_count=frozen, elapsed_seconds=float(r['elapsed_seconds']),
                            epochs_completed=int(r['epochs_completed']), best_epoch=int(r['best_epoch']),
                            checkpoint_bytes=(out/'best_validation.pt').stat().st_size,
                            physical_gpu=r.get('physical_gpu'), result_path=str(path), result_sha256=sha(path),
                            checkpoint_sha256=r['checkpoint_sha256'], independent_verification=verification.get(r['checkpoint_sha256']))
    result = dict(model=name, scope='stage2' if owner == STAGE else 'root', spec=spec,
                  complete=not pending, pending_seeds=pending, completed_seeds=[s for s in SEEDS if s in values],
                  per_seed={str(s): v for s, v in values.items()}, all_completed_seeds_independently_verified=all(v['independent_verification'] for v in values.values()) if values else False)
    if not pending:
        require(len({v['parameter_count_total'] for v in values.values()}) == 1, f'{name}: consistent architecture count across seeds')
        require(len({v['parameter_count_trainable'] for v in values.values()}) == 1, f'{name}: consistent trainable count across seeds')
        result['validation'] = {k: stats([values[s]['metrics'][k] for s in SEEDS]) for k in METRICS}
        result['composite'] = stats([values[s]['composite'] for s in SEEDS])
        result['cost'] = {k: stats([values[s][k] for s in SEEDS]) for k in ('elapsed_seconds', 'epochs_completed', 'best_epoch', 'checkpoint_bytes')}
        result['cost'].update(parameter_count_total=values[42]['parameter_count_total'], parameter_count_trainable=values[42]['parameter_count_trainable'],
                              explicit_frozen_parameter_count=values[42]['explicit_frozen_parameter_count'],
                              all_three_runs_elapsed_seconds=sum(values[s]['elapsed_seconds'] for s in SEEDS),
                              count_provenance='Reported model parameter count minus explicit frozen V3 control tensors; source-audited requires_grad policy; no model instantiated')
    return result


def compare_to(candidate, reference):
    if not candidate['complete'] or not reference['complete']:
        return dict(status='PENDING')
    paired = {k: stats([candidate['per_seed'][str(s)]['metrics'][k]-reference['per_seed'][str(s)]['metrics'][k] for s in SEEDS]) for k in METRICS}
    return dict(status='COMPLETE', deltas=paired, category=category({k: paired[k]['mean'] for k in METRICS}))


def mean_metrics(model):
    return {k: model['validation'][k]['mean'] for k in METRICS}


def ranking_key(model):
    return (CATEGORY_ORDER[model['versus_primary_BP']['category']], -model['validation']['f1_macro']['mean'],
            -model['composite']['mean'], model['cost']['parameter_count_trainable'])


def family_mapping(family_name, family_atoms, registry):
    eligible = {name: spec for name, spec in registry.items() if spec.get('kind') != 'mechanism_control' and spec.get('control', 'learned') == 'learned'}
    lookup = {}
    for size in range(4):
        for subset in itertools.combinations(family_atoms, size):
            key = frozenset(subset)
            matches = [name for name, spec in eligible.items() if atoms(spec) == key]
            require(len(matches) == 1, f'{family_name} subset {subset}: unique version-exact configuration needed, got {matches}')
            lookup[key] = matches[0]
    require(lookup[frozenset(family_atoms)] == family_name and lookup[frozenset()] == 'BP_control', 'Exact family full/control mapping')
    return lookup


def calculate_family(name, family_atoms, lookup, models):
    configs = []
    for subset, model_name in lookup.items():
        model = models[model_name]
        configs.append(dict(modules=sorted(subset), model=model_name, complete=model['complete'], pending_seeds=model['pending_seeds'],
                            validation=model.get('validation'), cost=model.get('cost'),
                            category_vs_primary_BP=model.get('versus_primary_BP', {}).get('category')))
    report = dict(family=name, module_versions=family_atoms, configurations=configs,
                  complete=all(c['complete'] for c in configs), modules={},
                  pending=[dict(model=c['model'], seeds=c['pending_seeds']) for c in configs if not c['complete']])
    for module in family_atoms:
        others = [m for m in family_atoms if m != module]; contexts = []
        for size in range(3):
            for background in itertools.combinations(others, size):
                before = lookup[frozenset(background)]; after = lookup[frozenset(background) | {module}]
                a, b = models[after], models[before]
                available = [s for s in SEEDS if str(s) in a['per_seed'] and str(s) in b['per_seed']]
                per_seed = {str(s): {k: a['per_seed'][str(s)]['metrics'][k]-b['per_seed'][str(s)]['metrics'][k] for k in METRICS} for s in available}
                complete = len(available) == 3
                contexts.append(dict(background_modules=list(background), before=before, after=after, complete=complete,
                                     full_minus_one=len(background) == 2, pending_seeds=[s for s in SEEDS if s not in available],
                                     per_seed_deltas=per_seed,
                                     metric_deltas={k: stats([per_seed[str(s)][k] for s in SEEDS]) for k in METRICS} if complete else None,
                                     trainable_parameter_delta=a['cost']['parameter_count_trainable']-b['cost']['parameter_count_trainable'] if complete else None,
                                     observed_elapsed_seconds_delta=stats([a['per_seed'][str(s)]['elapsed_seconds']-b['per_seed'][str(s)]['elapsed_seconds'] for s in SEEDS]) if complete else None))
        loo = next(c for c in contexts if c['full_minus_one'])
        module_report = dict(contexts=contexts, full_minus_one=loo, four_context_mean_marginal=None,
                             four_context_mean_f1_positive=None, full_minus_one_mean_f1_positive=None)
        if all(c['complete'] for c in contexts):
            marginal = {k: stats([sum(ctx['per_seed_deltas'][str(s)][k] for ctx in contexts)/4 for s in SEEDS]) for k in METRICS}
            f1s = list(marginal['f1_macro']['per_seed'].values())
            module_report.update(four_context_mean_marginal=marginal,
                                 four_context_mean_f1_positive=marginal['f1_macro']['mean'] > 1e-12,
                                 full_minus_one_mean_f1_positive=loo['metric_deltas']['f1_macro']['mean'] > 1e-12,
                                 positive_mean_marginal_seed_count=sum(x > 1e-12 for x in f1s),
                                 positive_full_minus_one_seed_count=sum(x > 1e-12 for x in loo['metric_deltas']['f1_macro']['per_seed'].values()),
                                 mean_trainable_parameter_delta=sum(ctx['trainable_parameter_delta'] for ctx in contexts)/4)
        report['modules'][module] = module_report
    full = models[name]
    report['numeric_top3_gate'] = None
    if report['complete']:
        classification = full['versus_primary_BP']['category']
        all_mean_positive = all(m['four_context_mean_f1_positive'] for m in report['modules'].values())
        all_loo_positive = all(m['full_minus_one_mean_f1_positive'] for m in report['modules'].values())
        report.update(numeric_top3_gate=classification in ('Strong', 'Acceptable') and all_mean_positive,
                      full_category=classification, all_modules_positive_four_context_mean_f1=all_mean_positive,
                      all_modules_positive_full_minus_one_mean_f1=all_loo_positive,
                      qualification='REQUIRES_MECHANISM_AND_NOVELTY_REVIEW' if classification in ('Strong', 'Acceptable') and all_mean_positive else 'FAILS_NUMERIC_PROMOTION_RULE')
    report['mechanism_review_required'] = True
    report['known_scope_limitations'] = []
    if 'M3_v2' in family_atoms:
        report['known_scope_limitations'].append('Old M3 V2 has an unresolved direct current-FC value bypass; V3 evidence cannot repair the V2 claim.')
    if any(x in family_atoms for x in ('M4', 'M5')):
        report['known_scope_limitations'].append('M4/M5 are generic readout controls; performance alone does not establish a new scientific contribution.')
    return report


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=STAGE/'summary/ablation_evidence')
    args = parser.parse_args(); torch.set_num_threads(2)
    registry = read(STAGE/'registry.json'); base_registry = read(RUN/'registry.json'); protocol = read(RUN/'protocol.json')
    rule = STAGE/'audits/Final_Selection_Rule_20261009.md'
    verification = load_verification_index()
    primary = collect_model('BrainPrompt', RUN, base_registry['BrainPrompt'], protocol, verification)
    models = {name: collect_model(name, STAGE, spec, protocol, verification) for name, spec in registry.items()}
    for model in models.values():
        model['versus_primary_BP'] = compare_to(model, primary)
        model['versus_contemporary_control'] = compare_to(model, models['BP_control'])
    mappings = {name: family_mapping(name, family_atoms, registry) for name, family_atoms in FAMILIES.items()}
    families = {name: calculate_family(name, FAMILIES[name], mappings[name], models) for name in FAMILIES}
    complete = [m for m in models.values() if m['complete'] and m['versus_primary_BP']['status'] == 'COMPLETE']
    ranked = sorted([m for m in complete if m['spec'].get('kind') not in ('control', 'mechanism_control')], key=lambda m: (ranking_key(m), m['model']))
    for i, model in enumerate(ranked, 1):
        model['display_rank'] = i
        model['promotion_category_pass'] = model['versus_primary_BP']['category'] in ('Strong', 'Acceptable')
    ranked_triples = [m['model'] for m in ranked if m['model'] in FAMILIES and families[m['model']]['complete']]
    eligible_simpler = [m['model'] for m in ranked if len(m['spec']['modules']) < 3 and m['promotion_category_pass']]
    numeric_top3 = [m['model'] for m in ranked if m['model'] in FAMILIES and families[m['model']]['numeric_top3_gate']]
    all_complete = primary['complete'] and all(m['complete'] for m in models.values())
    all_verified = all(m['all_completed_seeds_independently_verified'] for m in models.values()) and primary['all_completed_seeds_independently_verified']
    prior_summary = STAGE/'summary/validation_summary.json'; crosschecks = []
    if prior_summary.is_file():
        for row in read(prior_summary)['models']:
            name = row['model'].removeprefix('stage2/')
            current = models.get(name) if row['model'].startswith('stage2/') else primary if name == 'BrainPrompt' else None
            if current is None or not current['complete']:
                continue
            for key in METRICS:
                require(abs(row[key+'_mean']-current['validation'][key]['mean']) < 1e-12, f'Existing summary {name}/{key} mismatch')
                require(abs(row[key+'_std']-current['validation'][key]['sample_sd']) < 1e-12, f'Existing summary {name}/{key} SD mismatch')
            crosschecks.append(name)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='COMPLETE' if all_complete else 'PENDING',
                  script_sha256=sha(__file__), registry_sha256=sha(STAGE/'registry.json'), rule_path=str(rule), rule_sha256=sha(rule),
                  test_data_read=False, inference_performed=False, selection_lock_created=False,
                  required_seeds=SEEDS, profile=PROFILE, primary_baseline=primary, models=models, families=families,
                  pending_models={name:m['pending_seeds'] for name,m in models.items() if not m['complete']},
                  all_registered_development_complete=all_complete, all_completed_artifacts_independently_verified=all_verified,
                  existing_summary_models_crosschecked=crosschecks,
                  display_candidate_ranking=[m['model'] for m in ranked], numeric_top3_candidates=numeric_top3,
                  eligible_simpler_ranking=eligible_simpler, illustrative_triple_ranking=ranked_triples,
                  provisional_best_simpler=eligible_simpler[0] if eligible_simpler else None,
                  provisional_leading_illustrative_triple=ranked_triples[0] if ranked_triples else None,
                  final_selection_made=False, final_selection_ready_for_root_review=all_complete and all_verified,
                  formulas=dict(context_delta='metric(S union {module}, seed) - metric(S, seed)',
                                mean_marginal='mean over 3 seeds of the equal mean over all 4 subsets of the other 2 modules',
                                mean_marginal_sd='sample SD (ddof=1) of the 3 seed-level four-context means',
                                full_minus_one='metric(A+B+C, seed) - metric(other 2 modules, seed)',
                                positivity='>1e-12; same zero convention as existing summary',
                                composite='0.20 Accuracy + 0.25 F1 + 0.20 Precision + 0.20 Recall + 0.15 AUC',
                                cost_tiebreak='fewer trainable parameters, not total parameters including frozen controls'),
                  limitations=['Numeric qualification is necessary, not sufficient; mechanism bypass/degeneration and documented scientific differences require root review.',
                               'Promotion rule explicitly orders Strong then Acceptable. Mixed then Weak is a reporting-only completion of the order for illustrative triples; neither passes promotion.',
                               'Mean marginal is a four-context equal-weight contrast, not full-minus-one and not Shapley weighting.',
                               'Three seeds share the same validation set. Contexts are correlated, not 12 independent experiments.',
                               'Wall-clock durations reflect recorded training/export on possibly different GPUs; not a controlled inference benchmark.',
                               'No recommendation is frozen while registered experiments or independent verification are incomplete.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'ablation_evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    rankrows=[]
    for model in ranked:
        row=dict(model=model['model'],display_rank=model['display_rank'],category=model['versus_primary_BP']['category'],promotion_category_pass=model['promotion_category_pass'],
                 composite_mean=model['composite']['mean'],trainable_parameters=model['cost']['parameter_count_trainable'],total_parameters=model['cost']['parameter_count_total'],
                 elapsed_seconds_mean=model['cost']['elapsed_seconds']['mean'],elapsed_seconds_sum=model['cost']['all_three_runs_elapsed_seconds'])
        row.update({k+'_mean': model['validation'][k]['mean'] for k in METRICS});row.update({k+'_sample_sd':model['validation'][k]['sample_sd'] for k in METRICS})
        rankrows.append(row)
    csv_write(args.output_dir/'candidate_ranking.csv',rankrows)
    configrows=[];marginalrows=[];contextrows=[]
    for family, evidence in families.items():
        for config in evidence['configurations']:
            for seed in SEEDS:
                record=models[config['model']]['per_seed'].get(str(seed))
                configrows.append(dict(family=family,modules='+'.join(config['modules']) or 'BP',model=config['model'],seed=seed,status='COMPLETE' if record else 'PENDING',
                                       **(record['metrics'] if record else {})))
        for module, contribution in evidence['modules'].items():
            for seed in SEEDS:
                mean=contribution['four_context_mean_marginal'];loo=contribution['full_minus_one']['metric_deltas']
                marginalrows.append(dict(family=family,module=module,seed=seed,
                                         four_context_mean_f1=mean['f1_macro']['per_seed'][str(seed)] if mean else None,
                                         full_minus_one_f1=loo['f1_macro']['per_seed'][str(seed)] if loo else None,
                                         mean_marginal_f1=mean['f1_macro']['mean'] if mean else None,
                                         mean_marginal_f1_sample_sd=mean['f1_macro']['sample_sd'] if mean else None,
                                         full_minus_one_mean_f1=loo['f1_macro']['mean'] if loo else None))
            for context in contribution['contexts']:
                for seed in SEEDS:
                    contextrows.append(dict(family=family,module=module,background='+'.join(context['background_modules']) or 'BP',
                                            before=context['before'],after=context['after'],seed=seed,is_full_minus_one=context['full_minus_one'],
                                            status='COMPLETE' if str(seed) in context['per_seed_deltas'] else 'PENDING',**context['per_seed_deltas'].get(str(seed),{})))
    csv_write(args.output_dir/'eight_configuration_per_seed.csv',configrows)
    csv_write(args.output_dir/'module_marginals_per_seed.csv',marginalrows)
    csv_write(args.output_dir/'four_context_deltas_per_seed.csv',contextrows)
    lines=['# Validation-only ablation evidence','',f"Status: {report['status']}; registered matrix complete: {all_complete}; independent verification current: {all_verified}.",'',
           'Mean marginal averages all four backgrounds equally within each seed, then averages the three seeds. Full-minus-one is reported separately. No selection lock or test access occurs.','']
    for name, evidence in families.items():
        lines += [f'## {name}', '', '| Module | Four-context mean F1 delta | Sample SD | Full-minus-one F1 delta |', '|---|---:|---:|---:|']
        for module, contribution in evidence['modules'].items():
            mean=contribution['four_context_mean_marginal'];loo=contribution['full_minus_one']['metric_deltas']
            lines.append(f"| {module} | "+(f"{mean['f1_macro']['mean']:+.6f} | {mean['f1_macro']['sample_sd']:.6f}" if mean else 'PENDING | PENDING')+' | '+(f"{loo['f1_macro']['mean']:+.6f}" if loo else 'PENDING')+' |')
        lines += ['', 'Pending: '+json.dumps(evidence['pending']), 'Numeric gate: '+str(evidence['numeric_top3_gate']), '']
    lines += ['Eligible simpler ranking: '+', '.join(eligible_simpler), '', 'Illustrative triple ranking (provisional until all complete): '+', '.join(ranked_triples), '',
              'Mechanism/novelty judgment is not automated. Generic readouts and old M3 current-FC bypass require explicit qualification.']
    (args.output_dir/'Ablation_Evidence.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],completed_seed_runs=sum(len(m['completed_seeds']) for m in models.values()),pending=report['pending_models'],
                          eligible_simpler=eligible_simpler,numeric_top3=numeric_top3,illustrative_triple_ranking=ranked_triples,output=str(args.output_dir)),indent=2))


if __name__=='__main__':
    main()
