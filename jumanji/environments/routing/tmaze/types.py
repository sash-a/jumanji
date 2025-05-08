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

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import chex
import jax

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    agent_positions: jax.Array  # (2, 2)
    agent_targets: jax.Array  # (2, 2)
    step_count: jax.typing.ArrayLike  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    agents_view: chex.Array  # (num_agents, grid_size, grid_size)
    action_mask: chex.Array  # (num_agents, 5)
    step_count: chex.Array  # ()
