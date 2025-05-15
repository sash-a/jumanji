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

from jumanji.environments.routing.tmaze2.env import TMaze

n = 4
env = TMaze(n, 6, 2)
key = jax.random.key(100)

state, ts = env.reset(key)
# print(state)
print(ts.observation.agents_view.shape)
# print(ts.observation.action_mask.shape)
# choose step
action = 5 + jnp.arange(n)
state, ts = env.step(state, action)
print(state)
print(ts.observation.action_mask.shape)

for i in range(6):
    action = jnp.ones(n, dtype=int)
    state, ts = env.step(state, action)
    print(ts.reward, ts.last())

for i in range(2):
    action = jnp.zeros(n, dtype=int)
    state, ts = env.step(state, action)
    print(ts.reward, ts.last())

print(state)
print(ts.reward)
print(ts.observation)
print(ts.last())
print(ts.observation.agents_view.shape)


# states = [state]
# for i in range(10):
#     actions = jnp.array([1, 1])
#     state, ts = env.step(state, actions)
#     states.append(state)
#
# env.animate(states, interval=250, save_path="tmaze.mp4")
