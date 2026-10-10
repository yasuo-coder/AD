from .brainnet_pro import BrainNetProDataset
from .cohort import full_cohort_blockers
from .labels import require_label_policy


def LoadData(config):
    blockers = full_cohort_blockers(config)
    if blockers:
        raise ValueError('Full ADNI is not ready; no subset fallback is allowed:\n- ' + '\n- '.join(blockers)
                         + '\nRun python3 scripts/audit_full_adni.py for the seven-batch inventory.')
    require_label_policy(config.get('label_policy'), 'data config')
    if config.get('format') == 'prepared_cohort_v1':
        from .prepared_cohort import PreparedCohortDataset
        return PreparedCohortDataset(config)
    return BrainNetProDataset(config)
