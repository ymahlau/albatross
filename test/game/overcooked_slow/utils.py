import unittest

from src.game.overcooked_slow.utils import object_from_int


class TestUtils(unittest.TestCase):
    def test_obj_from_int_none(self):
        obj = object_from_int(3, (4, 2))
        self.assertIsNone(obj)

    def test_obj_from_int_onion(self):
        obj = object_from_int(0, (4, 2))
        print(obj)
        self.assertEqual('onion', obj.name)

    def test_obj_from_int_dish(self):
        obj = object_from_int(1, (4, 2))
        print(obj)
        self.assertEqual('dish', obj.name)

    def test_obj_from_int_soup(self):
        obj = object_from_int(2, (4, 2))
        print(obj)
        self.assertEqual('soup', obj.name)