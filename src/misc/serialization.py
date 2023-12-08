import dataclasses
from dataclasses import dataclass
import importlib
from enum import Enum
from typing import Any


@dataclass
class TupleWrapper:
    data: list


def serialize_obj(obj) -> Any:
    if isinstance(obj, Enum):
        return {
            'value': obj.value,
            '__module__': obj.__class__.__module__,
            '__name__': obj.__class__.__name__,
        }
    elif isinstance(obj, list):
        return [
            serialize_obj(v) for v in obj
        ]
    elif isinstance(obj, tuple):
        wrapped = TupleWrapper(list(obj))
        return serialize_dataclass(wrapped)
    elif dataclasses.is_dataclass(obj):
        return serialize_dataclass(obj)
    return obj


def serialize_dataclass(c) -> dict:
    data_dict = {}
    for field in dataclasses.fields(c):
        k = field.name
        v = getattr(c, k)
        data_dict[k] = serialize_obj(v)
    ser_dict = {
        'data': data_dict,
        '__module__': c.__class__.__module__,
        '__name__': c.__class__.__name__,
    }
    return ser_dict


def deserialize_obj(data: Any) -> Any:
    if isinstance(data, dict) and '__module__' in data and '__name__' in data and 'data' in data:
        # dataclass conversion
        return deserialize_dataclass(data)
    elif isinstance(data, dict) and '__module__' in data and '__name__' in data and 'value' in data:
        # enum conversion
        m = importlib.import_module(data['__module__'])
        c = getattr(m, data['__name__'])
        return c[data['value']]
    elif isinstance(data, list):
        # convert every item in list
        return [deserialize_obj(v) for v in data]
    return data


def deserialize_dataclass(d: dict):
    attribute_dict = {}
    for k, v in d['data'].items():
        attribute = deserialize_obj(v)
        attribute_dict[k] = attribute
    m = importlib.import_module(d['__module__'])
    c = getattr(m, d['__name__'])
    if c == TupleWrapper:  # tuple deserialization
        return tuple(attribute_dict['data'])
    res = c(**attribute_dict)
    return res
