"""Explicit class order, averaging conventions and scan/subject evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def classification_metrics(truth, probability, num_classes):
    y, probability = np.asarray(truth), np.asarray(probability)
    if not len(y) or probability.shape != (len(y), num_classes):
        raise ValueError('Metrics need nonempty labels and aligned class probabilities')
    pred = probability.argmax(-1)
    classes = np.arange(num_classes)
    matrix = confusion_matrix(y, pred, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    metrics = dict(accuracy=float(accuracy_score(y,pred)), samples=len(y),
                   present_classes=np.unique(y).tolist(), confusion_matrix=matrix.tolist(),
                   confusion_matrix_convention='rows=true, columns=predicted',
                   balanced_accuracy=float(recall[support>0].mean()),
                   zero_division=0, class_order=classes.tolist())
    for average in ('macro','weighted','micro'):
        p,r,f,_ = precision_recall_fscore_support(y,pred,labels=classes,average=average,zero_division=0)
        metrics.update({f'precision_{average}':float(p),f'recall_{average}':float(r),f'f1_{average}':float(f)})
    metrics['roc_auc_macro'] = None
    if len(np.unique(y)) == num_classes:
        metrics['roc_auc_macro'] = float(roc_auc_score(y,probability[:,1])) if num_classes==2 else float(
            roc_auc_score(y,probability,multi_class='ovr',average='macro',labels=classes))
    metrics['per_class'] = []
    for k in classes:
        fp = int(matrix[:,k].sum()-matrix[k,k])
        tn = int(matrix.sum()-matrix[k,:].sum()-fp)
        metrics['per_class'].append(dict(class_index=int(k),support=int(support[k]),
            predicted_count=int((pred==k).sum()), precision=float(precision[k]),recall=float(recall[k]),
            f1=float(f1[k]),specificity=float(tn/(tn+fp)) if tn+fp else None))
    return metrics


def subject_metrics(ids, truth, probability, num_classes):
    groups = {}
    for i, identity in enumerate(ids):
        groups.setdefault(identity.split('/')[0],[]).append(i)
    truth, probability = np.asarray(truth), np.asarray(probability)
    ys, probabilities = [], []
    for indices in groups.values():
        labels = np.unique(truth[indices])
        if len(labels) != 1:
            return dict(available=False, reason='Longitudinal diagnoses differ; no subject target was defined')
        ys.append(int(labels[0])); probabilities.append(probability[indices].mean(0))
    result = classification_metrics(ys,probabilities,num_classes)
    result.update(available=True, aggregation='mean of held-out scan probabilities per subject')
    return result
