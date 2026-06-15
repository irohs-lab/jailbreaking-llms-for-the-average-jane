from .base import OnlineLearningConfig
from . import rwm,exp3,linear_context,linucb, square_cb, uniform_priors, bcbf, thompson_sampling


def get_ol_algo_name(ol_cfg:OnlineLearningConfig):
    if ol_cfg.name == 'square_cb':
        return square_cb.get_sqcb_name(ol_cfg)
    else:
        return ol_cfg.name