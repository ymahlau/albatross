import dataclasses
from dataclasses import dataclass
import yaml
import importlib

from src.misc.serialization import serialize_dataclass, deserialize_dataclass


@dataclass
class TestClass:
    x: int
    y: float


@dataclass
class TestClass2:
    x: int
    y: float


@dataclass
class NestedClass:
    t1: TestClass
    t2: TestClass2
    x2: int


def main():
    c1 = TestClass(5, 6.6)
    c2 = TestClass2(5, 7.5)
    n = NestedClass(c1, c2, 5)
    ser_dict = serialize_dataclass(n)
    obj_new = deserialize_dataclass(ser_dict)
    a = 1




if __name__ == '__main__':
    main()
