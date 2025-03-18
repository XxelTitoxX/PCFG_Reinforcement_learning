import logging

logger = logging.getLogger(__name__)


class Writer:
    def __init__(self, name: str, config_dict: dict, use_wandb: bool):
        self.writer = None
        if use_wandb:
            logger.info(f"Writer - using wandb")
            import wandb
            wandb.init(project="learning-pcfg", name=name, config=config_dict)
            self.writer = wandb

    def log(self, log_dict: dict, commit: bool = True):
        if self.writer is not None:
            self.writer.log(log_dict, commit=commit)
