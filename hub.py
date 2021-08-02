import model.quartznet as qn
import model.configs as quartznet_configs


def get_quartznet(feat_in, vocab_size, **kwargs):
    return qn.QuartzNet(model_config=quartznet_configs._quartznet5x5_config, feat_in=feat_in, vocab_size=vocab_size, **kwargs)
