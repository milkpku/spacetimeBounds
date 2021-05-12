from .vec_env import VecEnv, CloudpickleWrapper
from .subproc_vec_env import SubprocVecEnv

from .base import BaseEnv
from .spacetime import SpacetimeBoundsEnv
from .spacetime_reg import SpacetimeRegEnv
from .imitation import ImitationEnv
from .style_energy import StyleEnergyEnv
from .style_volume import StyleVolumeEnv
from .style_gram import StyleGramEnv

env_dict = {
    "spacetime": SpacetimeBoundsEnv,
    "spacetime_reg": SpacetimeRegEnv,
    "imitation": ImitationEnv,
    "style_energy": StyleEnergyEnv,
    "style_volume": StyleVolumeEnv,
    "style_gram": StyleGramEnv,
}

def make_env(env_name, env_args):
  if env_name in env_dict:
    return env_dict[env_name](**env_args)
  else:
    print("environment %s not implemented" % env_name)
    raise NotImplementedError

def make_vec_env(env_name, env_args, env_num):
  if env_name not in env_dict:
    print("environment %s not implemented" % env_name)
    raise NotImplementedError

  env_class = env_dict[env_name]
  import numpy as np
  random_seeds = np.random.randint(0, 10000, env_num)
  vec_env = SubprocVecEnv([lambda : env_class(seed=x, **env_args) for x in random_seeds])

  return vec_env

