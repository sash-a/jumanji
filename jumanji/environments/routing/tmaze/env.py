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

from functools import cached_property
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import Environment, specs
from jumanji.environments.routing.tmaze.types import Observation, State
from jumanji.types import TimeStep, restart, termination, transition


class TMaze(Environment):
    def __init__(self, length: int, width: int, time_limit: int = 20) -> None:
        self.time_limit = time_limit
        super().__init__()

        self.length = length
        self.width = width  # only the width of the one side of the T

        self.left_target = jnp.array([self.length, -self.width])
        self.right_target = jnp.array([self.length, 1 + self.width])

        self.start_positions = jnp.array([[0, 0], [0, 1]])
        self.target_positions = jnp.stack([self.left_target, self.right_target], axis=0)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        key, position_key, target_key = jax.random.split(key, 3)

        a0_pos_idx = jax.random.randint(position_key, (), 0, 2)
        a0_pos = self.start_positions[a0_pos_idx]
        a1_pos = self.start_positions[1 - a0_pos_idx]

        a0_target_idx = jax.random.randint(target_key, (), 0, 2)
        a0_target = self.target_positions[a0_target_idx]
        a1_target = self.target_positions[1 - a0_target_idx]

        positions = jnp.stack([a0_pos, a1_pos], axis=0)
        targets = jnp.stack([a0_target, a1_target], axis=0)
        # TODO:
        action_mask = jnp.zeros((2, 6), dtype=bool).at[:, 4].set(True).at[:, 5].set(True)

        state = State(
            agent_positions=positions,
            agent_targets=targets,
            step_count=jnp.zeros((), jnp.int32),
            key=key,
        )
        obs = Observation(self.get_obs(state), action_mask, jnp.zeros((), jnp.int32))

        return state, restart(obs)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        # UP, RIGHT, DOWN, LEFT
        possible_moves = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        moves = possible_moves[action]
        new_positions = moves + state.agent_positions

        not_colliding = jnp.any(new_positions[0] != new_positions[1])
        valid_move = jax.vmap(self.valid_position, (None, 0))(state, new_positions) & not_colliding
        new_positions = jnp.where(valid_move[:, jnp.newaxis], new_positions, state.agent_positions)
        # TODO:
        action_mask = jnp.ones((2, 6), dtype=bool).at[:, 4].set(False).at[:, 5].set(False)

        new_state = State(
            agent_positions=new_positions,
            agent_targets=state.agent_targets,
            step_count=state.step_count + 1,
            key=state.key,
        )
        obs = Observation(self.get_obs(new_state), action_mask, new_state.step_count)

        reward = jnp.all(new_positions == state.agent_targets, axis=-1).astype(jnp.float32)

        done_horizon = new_state.step_count >= self.time_limit
        found_targets = jnp.all(reward == jnp.array([1.0, 1.0]))

        ts = jax.lax.cond(done_horizon | found_targets, termination, transition, reward, obs)
        return new_state, ts

    def get_obs(self, state: State) -> jax.Array:
        a0_obs = self.get_agent_obs(state, state.agent_positions[0])
        a1_obs = self.get_agent_obs(state, state.agent_positions[1])

        target_obs = jax.lax.cond(
            state.step_count == 0,
            lambda: jnp.all((state.agent_targets == self.left_target), axis=-1).astype(jnp.int32),
            lambda: jnp.array([-1, -1]),
        )

        obs = jnp.stack([a0_obs, a1_obs], axis=0)
        obs = jnp.concatenate([obs, target_obs[:, jnp.newaxis]], axis=-1)
        return obs

    def get_agent_obs(self, state: State, agent_position: jax.Array) -> jax.Array:
        surrounding_cell_values = self.surrounding_points(agent_position)
        return jax.vmap(self.get_cell_value, (None, 0))(state, surrounding_cell_values)

    def get_cell_value(self, state: State, cell_pos: jax.Array) -> jax.Array:
        in_bounds = self.is_cell_in_bounds(cell_pos)
        # -1 if out of bounds
        # 1 if agent 1
        # 2 if agent 2
        # 0 if empty cell
        return (
            (-1 * ~in_bounds)
            + (1 * jnp.all(cell_pos == state.agent_positions[0], axis=-1))
            + (2 * jnp.all(cell_pos == state.agent_positions[1], axis=-1))
        )

    def is_cell_in_bounds(self, cell_pos: jax.Array) -> bool:
        x, y = cell_pos
        is_on_vertical = (x >= 0) & (x < self.length)
        is_on_horizontal = x == self.length

        return (is_on_vertical & ((y == 0) | (y == 1))) | (
            is_on_horizontal & (y >= -self.width) & (y <= 1 + self.width)
        )

    def valid_position(self, state: State, new_position: jax.Array) -> bool:
        return (  # type: ignore
            self.is_cell_in_bounds(new_position)
            # not on top of an agent
            & jnp.any(new_position != state.agent_positions, axis=1).all()
        )

    def surrounding_points(self, cell_pos: jax.Array) -> jax.Array:
        surrounding_vecs = jnp.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, -1],
                [-1, 0],
                [-1, -1],
                [1, -1],
                [-1, 1],
            ]
        )
        return cell_pos + surrounding_vecs

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        agents_view = specs.BoundedArray(
            shape=(2, 10), dtype=jnp.int32, name="grid", minimum=-1, maximum=2
        )
        action_mask = specs.BoundedArray(
            shape=(2, 4), dtype=bool, minimum=False, maximum=True, name="action_mask"
        )
        step_count = specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.time_limit,
            name="step_count",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        return specs.MultiDiscreteArray(
            num_values=jnp.array([5] * 2), dtype=jnp.int32, name="action"
        )
