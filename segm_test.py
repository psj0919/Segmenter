import click
import torch
from Config.config_test_segm import get_config_dict
from core.engine_test_segm import Trainer




if __name__=='__main__':
    cfg = get_config_dict()
    trainer = Trainer(cfg)
    trainer.test()
