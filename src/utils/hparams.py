"""
Standalone hyperparameter loading from YAML files.
"""

import yaml


class HParams:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.__dict__:
                self.__dict__[key] = value

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
        return self

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __repr__(self):
        return '\nHyperparameters:\n' + '\n'.join(
            [f' {k}={v}' for k, v in self.__dict__.items()]
        )

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.load(f, Loader=yaml.SafeLoader))
