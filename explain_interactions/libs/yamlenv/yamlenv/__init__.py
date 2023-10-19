import typing as T
import yaml

from explain_interactions.libs.yamlenv.yamlenv import env, loader
from explain_interactions.libs.yamlenv.yamlenv.types import Stream


__version__ = "0.7.1"


def join(_loader, node):
    seq = _loader.construct_sequence(node)
    return "".join([str(_i) for _i in seq])


def load(stream):
    # type: (Stream) -> T.Any
    yaml.add_constructor("!join", join, loader.Loader)
    data = yaml.load(stream, loader.Loader)
    return env.interpolate(data)


def load_all(stream):
    # type: (Stream) -> T.Iterator[T.Any]
    for data in yaml.load_all(stream, loader.Loader):
        yield env.interpolate(data)
