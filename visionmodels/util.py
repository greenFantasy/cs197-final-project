import os

def get_pretrained_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "pretrained")

def modify_moco_dict(moco_state_dict):
    state_dict = moco_state_dict
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            # state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        elif 'encoder_k' in k or 'module.queue' in k:
            del state_dict[k]
        elif k.startswith('module.encoder_q.fc'):
            del state_dict[k]