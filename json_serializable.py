import json
from types import SimpleNamespace as Namespace
from enum import Enum


class JsonSerializable:

    @staticmethod
    def json_default(object):
        if isinstance(object, Enum):
            return object.name
        return object.__dict__

    def serialize(self, path):
        with open(path + "_" + self.__class__.__name__ + ".json", "w") as fp:
            json.dump(self, fp, default=JsonSerializable.json_default)

    @staticmethod
    def deserialize(path):
        with open(path + ".json", "r") as fp:
            return json.load(fp, object_hook=lambda d: Namespace(**d))

