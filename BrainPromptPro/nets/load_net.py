from .brainprompt_pro_net import BrainPromptProNet


def gnn_model(name, config, roi_prompt, disease_prompt, subject_prompt_dim, num_sites):
    if name != 'BrainPromptPro':
        raise ValueError(f'Unknown model {name}')
    profile = config.get('stability_profile', 'legacy')
    if profile == 'stable_v1':
        from .stable_brainprompt_pro_net import StableBrainPromptProNet
        return StableBrainPromptProNet(config, roi_prompt, disease_prompt, subject_prompt_dim, num_sites)
    if profile != 'legacy':
        raise ValueError(f'Unknown stability_profile {profile}')
    return BrainPromptProNet(config, roi_prompt, disease_prompt, subject_prompt_dim, num_sites)
