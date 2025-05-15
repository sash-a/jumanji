# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

from jumanji.environments.routing.switchgame.env import SwitchGameEnv

key = jax.random.key(10)
env = SwitchGameEnv(5, 10)
state, ts = env.reset(key)
print(state)
print(ts)
action = jnp.ones((5, 1))
state, ts = env.step(state, action)
print(state, ts)

print(action.at[1:5].set(0), jnp.array([1, 0, 0, 0, 0]))
state, ts = env.step(state, action.at[1:5].set(0))
print(state, ts)
