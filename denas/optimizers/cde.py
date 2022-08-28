import numpy as np
import ConfigSpace

from denas.optimizers.de import DE

class CatDE(DE):
    def __init__(self, cs: ConfigSpace.ConfigurationSpace, **kwargs):
        self.params = cs.get_hyperparameters()
        self.dim_map, num_new_params = self._gen_dim_map()
        super().__init__(
            cs=cs, dimensions=num_new_params, dim_map=self.dim_map, **kwargs)

    def _gen_dim_map(self):
        dim_map = {}
        num_new_params = 0
        for idx, param in enumerate(self.params):
            if isinstance(param, ConfigSpace.CategoricalHyperparameter):
                dim_map[idx] = []
                for _ in range(len(param.choices)):
                    dim_map[idx].append(num_new_params)
                    num_new_params += 1
            else:
                dim_map[idx] = [num_new_params]
                num_new_params += 1
        return dim_map, num_new_params

    def vectoridx_to_configparam(self, vector, idx, param):
        if isinstance(param, ConfigSpace.CategoricalHyperparameter):
            return param.choices[
                np.argmax(np.take(vector, self.dim_map[idx]))]
        else:
            return super().vectoridx_to_configparam(vector, idx, param)

    def vector_to_configspace(self, vector):
        '''Converts numpy array to ConfigSpace object

        Works when self.cs is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        new_vector = [
            np.take(vector, idxs) for idxs in self.dim_map.values()]

        new_config = self.cs.sample_configuration()
        for idx, param in enumerate(self.params):
            new_config[param.name] = self.vectoridx_to_configparam(vector, idx, param)
        return new_config

