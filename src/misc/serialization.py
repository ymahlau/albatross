import dataclasses
import importlib


def serialize_dataclass(c) -> dict:
    data_dict = {}
    for field in dataclasses.fields(c):
        k = field.name
        v = getattr(c, k)
        if dataclasses.is_dataclass(v):
            data_dict[k] = serialize_dataclass(v)
        else:
            data_dict[k] = v
    ser_dict = {
        'data': data_dict,
        '__module__': c.__class__.__module__,
        '__name__': c.__class__.__name__,
    }
    return ser_dict


def deserialize_dataclass(d: dict):
    data_dict = {}
    for k, v in d['data'].items():
        if isinstance(v, dict) and '__module__' in v and '__name__' in v:
            data_dict[k] = deserialize_dataclass(v)
        else:
            data_dict[k] = v
    m = importlib.import_module(d['__module__'])
    c = getattr(m, d['__name__'])
    res = c(**data_dict)
    return res
