import argparse
from explain_interactions.utils import read_json_or_yml, load_handler, load_dataset
from explain_interactions.registry import GRAPH_EXTRACTOR_REGISTRY
from explain_interactions.datasets import *  # noqa
from explain_interactions.models import *  # noqa
from explain_interactions.dataloaders import *  # noqa
from explain_interactions.tokenizers import *  # noqa
from explain_interactions.handlers import *  # noqa
from explain_interactions.methods.graphs import *  # noqa


def main():
    parser = argparse.ArgumentParser("Extract a graph")
    parser.add_argument("--config", type=str, default="${EXPL_INTR_HOME}/configs/base.yml")
    parser.add_argument("--limit", default=-1, type=int)
    args = parser.parse_args()

    configs = read_json_or_yml(args.config)
    print("loading data")
    instances = load_dataset(configs["dataset"])
    print(f"{len(instances)} instances loaded, loading model, tokenizer, handler")
    handler = load_handler(configs)
    print("model, tokenizer, handler loaded, getting results")
    graph_extractor = GRAPH_EXTRACTOR_REGISTRY[configs["graph_extractor"]["name"]](
        instances=instances, handler=handler, **configs["graph_extractor"].get("init_params", {})
    )
    extracted_graphs = graph_extractor.run(configs["graph_extractor"].get("run_params", {}))
    if "output" in configs["graph_extractor"]:
        extracted_graphs.save(configs["graph_extractor"]["output"])


if __name__ == "__main__":
    main()
