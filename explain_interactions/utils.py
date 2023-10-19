import os
import numpy as np
import torch
import json
import pandas as pd
from explain_interactions.libs.yamlenv import yamlenv
from typing import Any, Dict, List, Union, Optional, Tuple
from explain_interactions.datamodels import Instance, TxTokenizedInstance

from explain_interactions.registry import (
    DATASET_REGISTRY,
    TOKENIZER_REGISTRY,
    DATA_LOADER_REGISTRY,
    MODEL_REGISTRY,
    HANDLER_REGISTRY,
)


YML_EXTS = ["yml", "yaml"]


def str2bool(x):
    if x.lower() == "true":
        return True
    return False


def conv_path(_inp: str) -> str:
    """
    so that we can use the HOME dir
    :param _inp:
    :return:
    """
    return _inp.replace("${EXPL_INTR_HOME}", os.environ["EXPL_INTR_HOME"])


def conv_path_rev(_inp: str) -> str:
    return _inp.replace(os.environ["EXPL_INTR_HOME"], "${EXPL_INTR_HOME}")


def read_json_or_yml(path):
    path = conv_path(path)
    if path.endswith("json"):
        return json.load(open(path))
    elif any([path.endswith(ext) for ext in YML_EXTS]):
        return yamlenv.load(open(path))
    raise RuntimeError("config must be json or yaml")


def load_handler(configs: Dict):
    """
    :param configs:
    :return: a Handler
    """
    tokenizer = TOKENIZER_REGISTRY[configs["tokenizer"]]()
    data_loader = DATA_LOADER_REGISTRY[configs["data_loader"]](batchsz=16)
    model = MODEL_REGISTRY[configs["model"]].load()
    id_2_label = configs["dataset"]["id_2_label"]
    return HANDLER_REGISTRY[configs["handler"]](
        model=model, tokenizer=tokenizer, data_loader=data_loader, id_2_label=id_2_label
    )


def data_collator_with_str(features: List[Any]) -> Dict[str, Any]:
    first = features[0]
    batch = {}
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        elif not isinstance(v, str):
            batch[k] = torch.tensor([f[k] for f in features])
        else:
            batch[k] = [f[k] for f in features]
    return batch


def load_dataset(config: Dict) -> Union[List[Instance], List[TxTokenizedInstance]]:
    """
    load dataset from a config
    :param config:
    :return:
    """
    return DATASET_REGISTRY[config["name"]](**config).load(**config)


def results_to_csv(
    results: Dict[str, Dict[str, float]], class_labels: Optional[Dict[int, str]] = None, sep: str = "$"
) -> pd.DataFrame:
    """
    results is like this: {"auc-suff":{"ann_id:relation:level:class_label_index": value},
    "auc-comp":{"ann_id:relation:level:class_label_index": value}}. We assume each of the value dicts have the same number
    of keys.
    we will produce something like this:
    |class         | ann_id | relation | level | auc-suff | auc-comp |
    |----------------------------------------------------------------|
    |contradiction | 1      | synonym  | high  | 2.5      | 1.2      |
    |contradiction | 1      | synonym  | low   | 1.5      | 1.2      |
    |contradiction | 1      | antonym  | high  | 2.5      | 1.2      |
    |contradiction | 1      | antonym  | low   | 1.5      | 1.2      |
     :param results
     :param class_labels
     :param sep
    :return:{class_label: {annotator_id: {level: {relation: }}}} {}
    """
    row_headers = []
    row_values = np.zeros((len(list(results.values())[0]), len(results)))
    class_labels = {str(k): v for k, v in class_labels.items()} if class_labels is not None else None
    for index, result in enumerate(results.values()):
        for _index, (k, v) in enumerate(result.items()):
            if index == 0:
                ann_id, relation, level, class_label_index = k.split(sep)
                class_name = class_labels[class_label_index] if class_labels is not None else class_label_index
                row_headers.append([class_name, ann_id, relation, level])
            row_values[_index, index] = v
    final_list = list(np.hstack((row_headers, row_values)))
    df = pd.DataFrame(final_list, columns=["class", "ann", "relation", "level"] + list(results.keys()))
    return df.sort_values(by=["class", "ann", "relation", "level"])


def combine_dfs(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(results.values(), axis=1, keys=results.keys())


def create_exp_config_single_exp(all_config: Dict, exp_config: Dict) -> Tuple[bool, Dict]:
    """
    Get all necessary infor from a single experiment config, to be used for logging.
    :param all_config:
    :param exp_config:
    :return:
    """
    if not exp_config.get("log", True):
        return False, {}
    explainer_name = exp_config["explainers"]["base"]
    explainer_params = all_config["explainers"][explainer_name]
    explainer_params["scorer"] = all_config["scorers"][explainer_params["scorer"]]
    explainer_params["handler"] = all_config["handlers"][explainer_params["handler"]]

    dataset_name = exp_config["dataset"]
    dataset_params = all_config["datasets"][dataset_name]
    dataset_params["file_loc"] = conv_path_rev(dataset_params["file_loc"])

    evaluator_params = []
    for evaluator in exp_config["evaluators"]:
        evaluator_params.append(
            {all_config["evaluators"][evaluator]["label"]: all_config["evaluators"][evaluator].get("top_k", -1)}
        )

    exp_params = {
        "name": exp_config["name"],
        "dataset": dataset_params,
        "explainer": explainer_params,
        "top_k": evaluator_params,
    }
    return True, exp_params
