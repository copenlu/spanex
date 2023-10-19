import argparse
from typing import Dict, List, Tuple  # noqa
import os
import pandas as pd
from explain_interactions.datasets import *  # noqa
from explain_interactions.eval import *  # noqa
from explain_interactions.handlers import Handler
from explain_interactions.methods.expl_scorers import Scorer
from explain_interactions.eval.perturbations import Perturber
from explain_interactions.datamodels import Instance, InstanceExplanation, ComparativeExplanation, InstanceTextExplanation
from explain_interactions.reporting import WandbLogger
from explain_interactions.registry import (
    DATASET_REGISTRY,
    EXPL_SCORER_REGISTRY,
    HANDLER_REGISTRY,
    EXPL_METHOD_REGISTRY,
    EVAL_METHOD_REGISTRY,
    PERTURBATION_METHOD_REGISTRY,
)
from explain_interactions.utils import read_json_or_yml, results_to_csv, combine_dfs, create_exp_config_single_exp
from pprint import pprint
from copy import deepcopy
import json
import serpyco
from tqdm import tqdm

INST_TEXT_EXPL_SER = serpyco.Serializer(InstanceTextExplanation)


def load_dataset(dataset_params_: Dict) -> List[Instance]:
    dataset = DATASET_REGISTRY[dataset_params_["name"]](**dataset_params_)
    return dataset.load(**dataset_params_)


def load_handler(handler_params_: Dict) -> Handler:
    return HANDLER_REGISTRY[handler_params_["name"]].load(**handler_params_)


def load_scorer(scorer_params_: Dict, handlers_: Dict[str, Handler]) -> Scorer:
    handler_name = scorer_params_.pop("handler")
    handler = handlers_.get(handler_name)
    return EXPL_SCORER_REGISTRY[scorer_params_["name"]](handler=handler, **scorer_params_)


def load_explainer(explainer_params_: Dict, handlers_: Dict[str, Handler], scorers_: Dict[str, Scorer]) -> Scorer:
    scorer_name = explainer_params_.pop("scorer")
    scorer = scorers_.get(scorer_name)
    handler_name = explainer_params_.pop("handler")
    handler = handlers_.get(handler_name)
    print(f"loading explainer: {explainer_params_['name']}")
    return EXPL_METHOD_REGISTRY[explainer_params_["name"]](handler=handler, scorer=scorer, **explainer_params_)


def load_evaluator(evaluator_params_: Dict, handlers_: Dict[str, Handler], perturbers_: Dict[str, Perturber]):
    handler_name = evaluator_params_.pop("handler", None)
    handler = handlers_.get(handler_name) if handler_name is not None else None
    perturber_name = evaluator_params_.pop("perturber", None)
    perturber = perturbers_.get(perturber_name) if perturber_name is not None else None
    return EVAL_METHOD_REGISTRY[evaluator_params_["name"]](handler=handler, perturber=perturber, **evaluator_params_)


def load_perturber(perturber_params_: Dict, handlers_: Dict[str, Handler]):
    handler_name = perturber_params_.pop("handler", None)
    handler = handlers_.get(handler_name) if handler_name is not None else None
    return PERTURBATION_METHOD_REGISTRY[perturber_params_["name"]](handler=handler, **perturber_params_)


def run_comparative_experiment(exp_config_: Dict):
    """
    comparative evaluation of two types of explanations
    TODO: I don't think this is updated to handle the latest changes.
    :param exp_config_:
    :return:
    """
    _instances = instances[exp_config_["instances"]]
    print(f"running experiment {exp_config_['name']} using {len(_instances)} instances")
    explainer_base = explainers[exp_config_["explainers"]["base"]]
    explainer_alternate = explainers[exp_config_["explainers"]["alternate"]]
    explanations_base: Dict[str, InstanceExplanation] = {x.instance_idx: x for x in explainer_base.run(instances=_instances)}
    print("base explanations generated")
    explanations_alternate: Dict[str, InstanceExplanation] = {
        x.instance_idx: x
        for x in explainer_alternate.run(instances=_instances, instance_explanations=explanations_base.values())
    }
    print("alternate explanations generated")
    comparative_explanations = [
        ComparativeExplanation(instance_idx=k, base=v, alternate=explanations_alternate[k])
        for k, v in explanations_base.items()
    ]
    for evaluator_name in exp_config_["evaluators"]:
        result = evaluators[evaluator_name].run(comparative_explanations)
        pprint(evaluator_name, result)


def run_single_experiment(
    all_config: Dict, exp_config_: Dict, class_labels: Dict[int, str], all_explanations_: Dict[str, List[InstanceExplanation]]
) -> Dict:
    """
    single evaluation of an explanation
    :param all_config:
    :param exp_config_:
    :param class_labels
    :param all_explanations_: Useful for running explainers that use existing explanations
    :return:
    """
    expl_save_loc = exp_config_.get("expl_save_loc", None)
    should_log, exp_config_for_log = create_exp_config_single_exp(all_config, exp_config_)
    logger = WandbLogger(exp_name=exp_config_["name"], exp_config=exp_config_for_log) if should_log else None
    _instances: List[Instance] = instances[exp_config_["dataset"]]
    print(f"running experiment {exp_config_['name']} using {len(_instances)} instances")
    explainer = explainers[exp_config_["explainers"]["base"]]
    explanations: Dict[str, InstanceExplanation] = {
        x.instance_idx: x
        for x in explainer.run(
            instances=_instances, instance_explanations=all_explanations_.get(exp_config_.get("existing_explainer", "NA"), [])
        )
    }
    _instances = [x for x in _instances if x.idx in explanations]
    print(f"{len(explanations)} explanations generated")
    if expl_save_loc is not None:
        base_dir, file_name = os.path.split(expl_save_loc)
        os.makedirs(base_dir, exist_ok=True)
        with open(expl_save_loc, "w") as wf:
            _instance_dict: Dict[str, Instance] = {_instance.idx: _instance for _instance in _instances}
            for k, v in tqdm(explanations.items(), desc=f"Writing explanations to {expl_save_loc}"):
                _instance = _instance_dict[k]
                _i = InstanceTextExplanation(
                    instance_idx=_instance.idx,
                    label_index=_instance.label_index,
                    part1=_instance.part1,
                    part2=_instance.part2,
                    expl_phrases=sorted(v.expl_phrases, key=lambda x: x.score, reverse=True),
                )
                wf.write(json.dumps(INST_TEXT_EXPL_SER.dump(_i)) + "\n")
    all_results_ = {}
    if not exp_config_["evaluators"]:
        return {"results": pd.DataFrame(), "explanations": list(explanations.values())}
    for evaluator_name in exp_config_["evaluators"]:
        result = evaluators[evaluator_name].run(instances=_instances, explanations=list(explanations.values()))
        all_results_[evaluator_name] = result

    if type(list(all_results_.values())[0] == dict):
        all_results_ = results_to_csv(results=all_results_, class_labels=class_labels)
    if should_log:
        logger.done(all_results_)

    # pprint(all_results_.to_dict(orient="records"))
    return {"results": all_results_, "explanations": list(explanations.values())}


def get_class_labels_for_dataset(_exp_config: Dict, dataset_infos: Dict):
    dataset_info = dataset_infos[_exp_config["dataset"]]
    return dataset_info["id_2_label"]


parser = argparse.ArgumentParser("baselines")
parser.add_argument("--config_file", default="${EXPL_INTR_HOME}/configs/evaluate_single.yml")
parser.add_argument("--use_signac", default=False, action="store_true")
args = parser.parse_args()
all_configs = read_json_or_yml(args.config_file)
all_configs_dc = deepcopy(all_configs)
if args.use_signac:
    from yaml import safe_dump
    import signac

    project = signac.get_project("explintr")
    _configs = deepcopy(all_configs)
    job = project.open_job(_configs)
    job.init()
    safe_dump(all_configs, open(job.fn("config.yml"), "w"))

print("loading datasets")
instances = {k: load_dataset(v) for k, v in all_configs["datasets"].items()}
print("datasets loaded, loading handlers")
handlers = {k: load_handler(v) for k, v in all_configs["handlers"].items()}
print("handlers loaded, loading scorers")
scorers = {k: load_scorer(v, handlers_=handlers) for k, v in all_configs["scorers"].items()}
print("scorers loaded, loading explainers")
explainers = {k: load_explainer(v, scorers_=scorers, handlers_=handlers) for k, v in all_configs["explainers"].items()}
print("explainers loaded, loading perturbers")
perturbers = {k: load_perturber(v, handlers_=handlers) for k, v in all_configs["perturbers"].items()}
print("perturbers loaded, loading evaluators")
evaluators = {k: load_evaluator(v, handlers_=handlers, perturbers_=perturbers) for k, v in all_configs["evaluators"].items()}
print("evaluators loaded, running experiments")


all_results = {}
all_explanations = {}
for exp_name, exp_config in all_configs["experiments"].items():
    if exp_config["exp_type"] == "comparative":  # TODO: this is no longer supported
        run_comparative_experiment(exp_config)
    else:
        results_explanations = run_single_experiment(
            all_configs_dc, exp_config, get_class_labels_for_dataset(exp_config, all_configs["datasets"]), all_explanations
        )
        all_results[exp_name] = results_explanations["results"]
        all_explanations[exp_name] = results_explanations["explanations"]
        print(f"finished running {exp_name}")
if args.use_signac:
    final_df = combine_dfs(all_results)
    with open(job.fn("results.csv"), "w") as wf:
        final_df.to_csv(wf)
        print(f"results written to {job.fn('results.csv')}")
