"""Checkpoint interventions on M1 TRAIN-gradient/VALIDATION only; never test."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/panyao/AD')
import argparse
import json
from pathlib import Path
import types
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from BrainPromptPro import candidate_runner as r
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.data.brainnet_pro import FoldDataset, sha256
from BrainPromptPro.nets.research_candidates_v1 import normalize_nonnegative


def stats(x):
    x = x.detach().double()
    return dict(mean=float(x.mean()), std=float(x.std(unbiased=False)),
                min=float(x.min()), max=float(x.max()))


def constant_graph(self, fc, prompt):
    gates = torch.full((*fc.shape, 2), .625, dtype=fc.dtype, device=fc.device)
    mask = 1 - torch.eye(fc.shape[-1], dtype=fc.dtype, device=fc.device)
    return fc.clamp_min(0) * .625 * mask, (-fc).clamp_min(0) * .625 * mask, gates


def zero_message(self, h, pos, neg, index):
    return torch.zeros_like(h)


@torch.no_grad()
def infer(model, loader, capture=False):
    logits, truth, ids, gates, adj_delta = [], [], [], [], []
    scale_error, degree_min = 0., float('inf')
    for batch in loader:
        o = r.forward_batch(model, batch, 'cpu')
        logits.append(o['logits'].cpu()); truth.append(batch['label']); ids += batch['subject_id']
        if capture:
            gates.append(o['edge_gates'].cpu())
            fc = batch['fc']; mask = 1-torch.eye(fc.shape[-1])
            for name, raw in [('positive', fc.clamp_min(0)*mask), ('negative', (-fc).clamp_min(0)*mask)]:
                learned = normalize_nonnegative(o['signed_adjacencies'][name])
                fixed = normalize_nonnegative(raw*.625)
                adj_delta.append(dict(branch=name, mean_absolute=float((learned-fixed).abs().mean()),
                                      max_absolute=float((learned-fixed).abs().max()),
                                      relative_frobenius=float((learned-fixed).norm()/fixed.norm().clamp_min(1e-12))))
                positive_degree=raw.sum(-1); positive_degree=positive_degree[positive_degree>0]
                if len(positive_degree): degree_min=min(degree_min,float(positive_degree.min()))
                for factor in [.25,.625,1.]:
                    scale_error=max(scale_error,float((normalize_nonnegative(raw*factor)-normalize_nonnegative(raw)).abs().max()))
    x=torch.cat(logits); y=torch.cat(truth); p=x.softmax(-1)
    answer=dict(logits=x,probability=p,truth=y,sample_ids=ids)
    answer['metrics']=r.classification_metrics(y.numpy(),p.numpy(),model.num_classes)
    if capture:
        g=torch.cat(gates); mask=~torch.eye(g.shape[1],dtype=torch.bool); off=g[:,mask]
        answer['gate_diagnostics']=dict(offdiagonal=stats(off),
            positive=stats(off[...,0]),negative=stats(off[...,1]),
            across_patient_gate_std_mean=float(off.std(0,unbiased=False).mean()),
            gate_distance_from_0625=stats((off-.625).abs()),
            normalized_graph_learned_vs_fixed_batches=adj_delta,
            normalization_scale_invariance_max_absolute_error=scale_error,
            minimum_positive_raw_sign_degree=degree_min)
    return answer


def selected_gradient_probe(model, data, split, enc, checkpoint):
    batch=next(iter(DataLoader(FoldDataset(data,split['train'][:20],enc),batch_size=20)))
    model.zero_grad(set_to_none=True)
    o=r.forward_batch(model,batch,'cpu')
    main=F.cross_entropy(o['logits'],batch['label'],weight=checkpoint['class_weights'])
    aux=F.cross_entropy(o['disease_logits'],batch['label'],weight=checkpoint['class_weights'])
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params=[p for n,p in named]
    gm=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True)
    ga=torch.autograd.grad(aux,params,allow_unused=True)
    result={}
    for group,prefix in [('M1','adaptive.'),('edge','adaptive.edge.'),('fc_encoder','adaptive.fc_encoder.'),('roi_encoder','adaptive.roi_encoder.'),('signed_positive','adaptive.pos.'),('signed_negative','adaptive.neg.'),('message_output','adaptive.out.')]:
        idx=[i for i,(n,p) in enumerate(named) if n.startswith(prefix)]
        a=torch.cat([(gm[i] if gm[i] is not None else torch.zeros_like(params[i])).flatten() for i in idx])
        b=torch.cat([(ga[i] if ga[i] is not None else torch.zeros_like(params[i])).flatten() for i in idx])
        result[group]=dict(class_weighted_main_ce_gradient_norm=float(a.norm()),
            class_weighted_auxiliary_ce_gradient_norm_before_lambda=float(b.norm()),
            main_aux_cosine=float(F.cosine_similarity(a[None],b[None])),
            adam_weight_decay_vector_norm=float(checkpoint['config']['params']['weight_decay']*sum(float(params[i].detach().square().sum()) for i in idx)**.5))
    return dict(role='First 20 fixed TRAIN samples, eval mode, no optimizer update',groups=result)


def write_report(aggregate, path):
    lines=['# M1 V2 最终机制反事实审查（2026-10-09）','',
        '范围：三个已完成 seed 的最佳 validation 检查点；每次仅对固定 validation 100 个 acquisition 推理。另在固定 TRAIN 前20例上计算梯度，不更新权重。未创建 test loader、未运行 test 推理、未读取 test 结果。','',
        '## 检查点干预结果','',
        '| Seed | 原始 macro-F1 | 固定门 macro-F1 | 固定门预测变化/100 | 概率平均绝对变化 | 关闭消息 macro-F1 | 关闭消息预测变化/100 |','|---|---:|---:|---:|---:|---:|---:|']
    for z in aggregate['seeds']:
        a=z['treatments']['original']; b=z['treatments']['fixed_gates_0625']; c=z['treatments']['zero_signed_messages']
        lines.append(f"| {z['seed']} | {a['metrics']['f1_macro']:.9f} | {b['metrics']['f1_macro']:.9f} | {b['prediction_changes']} | {b['mean_absolute_probability_delta']:.8g} | {c['metrics']['f1_macro']:.9f} | {c['prediction_changes']} |")
    lines += ['','固定门把每条边的正/负门都替换为0.625，保留原始 signed FC、正负两路变换、归一化、LayerNorm与已训练输出；关闭消息将每层 adaptive.message 的输出置零，形状与原 hidden states 相同。两者都是原检查点上的反事实敏感性探针，**不是重新训练消融，不等同于独立训练的 BP_control，也不能单独作因果贡献证明**。','',
        '## 门分布与梯度','',
        '| Seed | 非对角 gate mean | 非对角 gate std | across-patient gate std | selected-checkpoint edge CE grad norm | M1整体 CE grad norm |','|---|---:|---:|---:|---:|---:|']
    for z in aggregate['seeds']:
        g=z['gate_diagnostics']; p=z['selected_train_gradient']['groups']
        lines.append(f"| {z['seed']} | {g['offdiagonal']['mean']:.9f} | {g['offdiagonal']['std']:.8g} | {g['across_patient_gate_std_mean']:.8g} | {p['edge']['class_weighted_main_ce_gradient_norm']:.8g} | {p['M1']['class_weighted_main_ce_gradient_norm']:.8g} |")
    lines += ['','JSON同时保留每个 seed 的首epoch训练梯度最大值、已有诊断内容（如存在）和本次统一 TRAIN20 梯度。首epoch逐参数最大值与最佳检查点组梯度定义不同，不混用或直接相减。分类和辅助CE均使用训练类权重；辅助CE梯度记录的是乘lambda_disease之前的值，weight-decay向量单列。','',
        '## 统一门尺度为何抵消','',
        '`normalize_nonnegative(A)=D(A)^(-1/2) A D(A)^(-1/2)`，每个正负分支独立计算非负度数。若门在某个分支所有边上均为常数 c>0，`D(cA)=cD(A)`，所以 `normalize_nonnegative(cA)=normalize_nonnegative(A)`。正负两支即使取不同正常数，也会各自在自己的度归一化中消失。只有相对边权变化能够改变传播算子。','',
        '数值例外是正度数触发1e-8 clamp；零度数行仍输出零。本次在全部 validation FC 的两支检查 c∈{0.25,0.625,1.0}：']
    for z in aggregate['seeds']:
        g=z['gate_diagnostics']; lines.append(f"- seed {z['seed']}：最大归一化矩阵差 {g['normalization_scale_invariance_max_absolute_error']:.8g}；最小正的原始sign-degree {g['minimum_positive_raw_sign_degree']:.8g}。")
    lines += ['','因此图随患者、类别变化不能直接归因于 learned gate：固定门仍保留每个患者的完整 signed FC差异；signed-message权重和输出可以有作用，而边门几乎保持常数。','',
        '## 证据支持的结论','',
        '结合逐seed固定门干预、接近常数的门、很小的患者间门变化及edge梯度，当前证据不支持“已经学到自适应疾病脑图”。M1应最多保留为**signed-message 保守实验候选**，性能归因应写为已训练正负数值FC消息分路及其残差，而不是疾病特异拓扑学习。关闭消息的敏感性只说明既有检查点依赖这条分支；若要确认结构泛化增益，仍需相同训练协议的固定门重训和去消息重训对照。','',
        'Plus overlap结论没有改变：本模块未把频谱或ROI-pair FC描述交给LLM，没有复现Plus spectral/spatial文本输入图；通过Plus排除不代表机制有效或领域首创。','',
        '## 可复现性与原件保护','',
        '- 脚本：`audit_scripts/review_m1_mechanism_20261009.py`。仅内存替换实例方法，finally恢复；不写训练源码、不更新模型参数或检查点。',
        '- 汇总JSON：`audits/M1_Final_Mechanism_Review_20261009.json`；逐seed JSON及逐患者概率：`audits/M1_mechanism_20261009/seed_*/`。',
        '- 对照原val_payload按ID和真值严格匹配，核对原始推理概率误差；检查点及参与前向实现的源码哈希在运行前后核验。',
        '- 使用CPU，两线程；三个seed统一batch_size=20、eval mode、相同validation顺序；无随机训练和无test访问。','']
    path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('/tmp/M1_mechanism_20261009'));args=parser.parse_args()
    dest=args.output_dir;dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cfg,protocol,data,split=r.load_context()
    assert len(split['val'])==100
    aggregate=dict(role='Validation checkpoint interventions / TRAIN gradient only',test_accessed=False,
        training_code_changed=False,optimizer_updates=0,device='cpu',threads=2,batch_size=20,
        split_sha256=protocol['split_sha256'],script_sha256=sha256(__file__),seeds=[])
    for seed in [42,3407,2026]:
        root=r.experiment_path('M1','sqrt_inverse_frequency',seed);checkpoint_path=root/'best_validation.pt'
        checkpoint_hash=sha256(checkpoint_path);ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert ck['training_indices_used']==split['train']
        checked={}
        for source,digest in ck['source_sha256'].items():
            # Verify all previously snapshotted training implementation, not new audit scripts.
            assert sha256(source)==digest, f'Source changed: {source}'
            checked[source]=digest
        enc=MetadataEncoder.from_state_dict(ck['metadata_encoder']);r.seed_all(seed)
        model,c=r.build_model('M1',ck['config'],data,enc,ck['spec']);model.load_state_dict(ck['model']);model.eval()
        loader=DataLoader(FoldDataset(data,split['val'],enc),batch_size=20,shuffle=False)
        original=infer(model,loader,capture=True)
        old=torch.load(root/'val_payload.pt',map_location='cpu',weights_only=True)
        assert old['sample_ids']==original['sample_ids'];assert torch.equal(old['truth'],original['truth'])
        reproduces_delta=float((old['probability']-original['probability']).abs().max())
        assert reproduces_delta<=1e-5,reproduces_delta
        graph_method=model.adaptive.graph;message_method=model.adaptive.message
        payloads={'original':original}
        try:
            model.adaptive.graph=types.MethodType(constant_graph,model.adaptive)
            payloads['fixed_gates_0625']=infer(model,loader)
            model.adaptive.graph=graph_method
            model.adaptive.message=types.MethodType(zero_message,model.adaptive)
            payloads['zero_signed_messages']=infer(model,loader)
        finally:
            model.adaptive.graph=graph_method;model.adaptive.message=message_method
        probe=selected_gradient_probe(model,data,split,enc,ck)
        trained=json.loads((root/'result.json').read_text())
        saved_diag=root/'diagnostics/module_diagnostics.json'
        report=dict(seed=seed,checkpoint_epoch=ck['epoch']+1,checkpoint_sha256=checkpoint_hash,
            role='Validation 100 acquisition checkpoint sensitivity; not retrained ablation',
            test_accessed=False,validation_payload_reproduction_max_probability_delta=reproduces_delta,
            gate_diagnostics=original['gate_diagnostics'],selected_train_gradient=probe,
            first_epoch_adaptive_gradient_norm_max={n:v for n,v in trained['gradient_norms_first_epoch'].items() if n.startswith('adaptive.')},
            existing_diagnostics=json.loads(saved_diag.read_text()) if saved_diag.exists() else None,
            treatments={})
        for name,value in payloads.items():
            assert value['sample_ids']==original['sample_ids'];assert torch.equal(value['truth'],original['truth'])
            delta=(value['probability']-original['probability']).abs()
            report['treatments'][name]=dict(metrics=value['metrics'],
                prediction_changes=int((value['probability'].argmax(-1)!=original['probability'].argmax(-1)).sum()),
                mean_absolute_probability_delta=float(delta.mean()),max_absolute_probability_delta=float(delta.max()),
                max_absolute_logit_delta=float((value['logits']-original['logits']).abs().max()))
        assert sha256(checkpoint_path)==checkpoint_hash
        assert all(sha256(source)==digest for source,digest in checked.items())
        report['checkpoint_and_source_unchanged_after_probe']=True
        out=dest/f'seed_{seed}';out.mkdir(exist_ok=True)
        (out/'interventions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raw=dict(sample_ids=original['sample_ids'],truth=original['truth'].tolist(),
            treatments={name:dict(probability=value['probability'].tolist(),predictions=value['probability'].argmax(-1).tolist()) for name,value in payloads.items()})
        (out/'validation_probabilities.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
        aggregate['seeds'].append(report)
        print(json.dumps(dict(seed=seed,treatments=report['treatments'],gate_std=report['gate_diagnostics']['offdiagonal']['std'])),flush=True)
    (dest/'M1_Final_Mechanism_Review_20261009.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
    write_report(aggregate,dest/'M1_Final_Mechanism_Review_20261009.md')


if __name__=='__main__':main()
