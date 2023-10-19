import hashlib
import json
import wandb
from typing import Dict
import getpass
import socket
import pandas as pd


class WandbLogger:
    def __init__(self, exp_name: str, exp_config: Dict, **kwargs):
        self.exp_name = exp_name
        self.exp_config = exp_config
        self.exp_config_hash = hashlib.md5(json.dumps(exp_config).encode("utf-8")).hexdigest()
        username = kwargs.get("user", getpass.getuser())
        hostname = kwargs.get("host", socket.gethostname())
        self.exp_config.update({"username": username, "hostname": hostname})
        self.exp_config.update({"config_hash": self.exp_config_hash})

    def done(self, result: pd.DataFrame):
        d = result.to_dict(orient="records")
        for index, item in enumerate(d):
            wandb.init(project=self.exp_name, config=self.exp_config, entity="explintr", reinit=True)
            item.update({"row": index})
            wandb.log(item)
        wandb.finish(exit_code=0)
