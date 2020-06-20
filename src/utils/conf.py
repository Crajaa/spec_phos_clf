from ruamel.yaml import YAML


class Params():
    def __init__(self, yaml_fn, config_name):
        self.dict_params = {}
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.dict_params[k] = v

    def get_dict_params(self):
        """
        """
        return self.dict_params

    def set_dict_params(self, new_params):
        """
        """
        try:
            assert(isinstance(new_params, dict))
            self.dict_params = new_params
        except AssertionError as err1:
            print(
                "Argument of set_dict_params method of Params class must be a \
                    dictionary. Received : " + str(type(new_params))
                )
            raise(err1)
