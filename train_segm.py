import click

from Config.config_segm import get_config_dict
from core.engine_segm import Trainer




if __name__=='__main__':
    cfg = get_config_dict()
    trainer = Trainer(cfg)
    trainer.training()
    # trainer.validation()

