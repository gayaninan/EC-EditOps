import yaml

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value) 
            self.__dict__[key] = value

    def __getattr__(self, key):
        return self.__dict__.get(key)
