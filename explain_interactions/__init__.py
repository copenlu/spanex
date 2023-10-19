# flake8: noqa
import logging
from datetime import datetime
import os

DT_FMT = "%Y%m%d:%H%M%S"

logger = logging.getLogger("explain-interactions")
os.makedirs(
    os.path.join(os.environ["EXPL_INTR_HOME"], "data/logs"), exist_ok=True
)
logger.addHandler(
    logging.FileHandler(
        os.path.join(
            os.environ["EXPL_INTR_HOME"],
            f"data/logs/explintr-{datetime.now().strftime(DT_FMT)}.log",
        )
    )
)
logger.setLevel(logging.DEBUG)
