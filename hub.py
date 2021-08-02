from .model.quartznet import QuartzNet
from .model.configs import _quartznet5x5_config


def get_quartznet(feat_in, vocab_size, **kwargs):
    return QuartzNet(model_config=_quartznet5x5_config, feat_in=feat_in, vocab_size=vocab_size, **kwargs)
