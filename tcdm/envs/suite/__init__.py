# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from tcdm.envs.wrappers import GymWrapper
from tcdm.envs.suite.tcdm import TCDM_DOMAINS
from tcdm.envs.suite.tcdm_multi import get_domain_multi


def load(env_name, task_kwargs=None, environment_kwargs=None, gym_wrap=False):

    _DOMAINS = TCDM_DOMAINS
    domain_name, task_name = env_name.split('-')
    if domain_name not in _DOMAINS:
        raise ValueError("Domain {} does not exist!".format(domain_name))

    domain = _DOMAINS[domain_name]
    if task_name not in domain:
        raise ValueError("Task {} does not exist in domain {}".format(task_name, domain_name))
    task_kwargs = {} if task_kwargs is None else task_kwargs
    environment_kwargs = {} if environment_kwargs is None else environment_kwargs
    task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
    env = domain[task_name](**task_kwargs)
    env = GymWrapper(env) if gym_wrap else env
    return env


def load_multi(env_name, task_kwargs=None, environment_kwargs=None, gym_wrap=False):
    domain = get_domain_multi(env_name, task_kwargs=task_kwargs)

    task_kwargs = {} if task_kwargs is None else task_kwargs
    environment_kwargs = {} if environment_kwargs is None else environment_kwargs
    task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
    task_name = env_name.split('_')[-1]
    env = domain[task_name](**task_kwargs)
    env = GymWrapper(env) if gym_wrap else env
    return env