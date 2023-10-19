from functools import wraps

DATASET_REGISTRY = {}
EVAL_METHOD_REGISTRY = {}
EXPL_METHOD_REGISTRY = {}
HANDLER_REGISTRY = {}
TOKENIZER_REGISTRY = {}
MODEL_REGISTRY = {}
DATA_LOADER_REGISTRY = {}
GRAPH_EXTRACTOR_REGISTRY = {}
EXPL_SCORER_REGISTRY = {}
PERTURBATION_METHOD_REGISTRY = {}
PLOTTER_DF_LOADER_REGISTRY = {}
PLOTTER_METHOD_REGISTRY = {}

DATASET = "dataset"
HANDLER = "handler"
EVAL_METHOD = "eval_method"
EXPL_METHOD = "expl_method"
TOKENIZER = "tokenizer"
MODEL = "model"
DATA_LOADER = "data_loader"
GRAPH_EXTRACTOR = "graph_extractor"
EXPL_SCORER = "explanation_scorer"
PERTURBATION_METHOD = "perturbation_method"
PLOTTER_DF_LOADER = "plotter_df_loader"
PLOTTER_METHOD = "plotter_method"


def optional_params(func):
    """Allow a decorator to be called without parentheses if no kwargs are given.

    parameterize is a decorator, function is also a decorator.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        """If a decorator is called with only the wrapping function just execute the real decorator.
           Otherwise return a lambda that has the args and kwargs partially applied and read to take a function as an argument.

        *args, **kwargs are the arguments that the decorator we are parameterizing is called with.

        the first argument of *args is the actual function that will be wrapped
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)

    return wrapped


@optional_params
def register(cls, _type: str, _name: str):
    if _type == DATASET:
        DATASET_REGISTRY[_name] = cls
    elif _type == EVAL_METHOD:
        EVAL_METHOD_REGISTRY[_name] = cls
    elif _type == EXPL_METHOD:
        EXPL_METHOD_REGISTRY[_name] = cls
    elif _type == HANDLER:
        HANDLER_REGISTRY[_name] = cls
    elif _type == TOKENIZER:
        TOKENIZER_REGISTRY[_name] = cls
    elif _type == MODEL:
        MODEL_REGISTRY[_name] = cls
    elif _type == DATA_LOADER:
        DATA_LOADER_REGISTRY[_name] = cls
    elif _type == GRAPH_EXTRACTOR:
        GRAPH_EXTRACTOR_REGISTRY[_name] = cls
    elif _type == EXPL_SCORER:
        EXPL_SCORER_REGISTRY[_name] = cls
    elif _type == PERTURBATION_METHOD:
        PERTURBATION_METHOD_REGISTRY[_name] = cls
    elif _type == PLOTTER_DF_LOADER:
        PLOTTER_DF_LOADER_REGISTRY[_name] = cls
    elif _type == PLOTTER_METHOD:
        PLOTTER_METHOD_REGISTRY[_name] = cls
    else:
        raise RuntimeError(f"No suitable registry found for type {_type}")
    return cls
