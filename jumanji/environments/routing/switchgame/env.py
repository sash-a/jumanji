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

from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.types import StepType, TimeStep, termination, transition

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class SwitchGameState:
    """
    State of the SwitchGame environment.

    Attributes:
        key: PRNG key for JAX.
        agent_int_states: A JAX array of integers (0 or 1) representing each agent's state.
                          Shape: (num_agents,).
        goal_sum: The target sum for the agent states. Scalar integer.
        step_count: Current step count in the episode. Scalar integer.
    """

    key: chex.PRNGKey
    agent_int_states: chex.Array
    goal_sum: jnp.int32
    step_count: jnp.int32


class Observation(NamedTuple):
    agents_view: chex.Array  # (num_agents, grid_size, grid_size)
    action_mask: chex.Array  # (num_agents, 5)
    step_count: chex.Array  # ()


class SwitchGameEnv(Environment[SwitchGameState, specs.BoundedArray, Observation]):
    """
    A multi-agent RL environment where agents control a boolean state (0 or 1).
    The goal is for the sum of agent states to match a randomly chosen target sum.

    Environment Details:
    - State (`SwitchGameState`):
        - key: PRNG key.
        - agent_int_states: jnp.int32 array of shape (num_agents,), stores 0 or 1 for each agent.
        - goal_sum: jnp.int32 scalar, the target sum.
        - step_count: jnp.int32 scalar, current episode step.

    - Actions (type `chex.Array`, spec `specs.BoundedArray`):
        - A jnp.int32 array of shape (num_agents,).
        - Each element is 0 (do nothing) or 1 (switch state).

    - Observation (type `chex.Array`, spec `specs.Array`):
        - A jnp.float32 array of shape (num_agents, 3).
        - Each row corresponds to an agent's observation:
            1. `my_current_state`: The agent's own current state (0.0 or 1.0).
            2. `sum_all_states_normalized`: Sum of all agent states / num_agents.
            3. `goal_sum_normalized`: The target sum / num_agents.

    - Reward (spec `specs.Array`):
        - Scalar jnp.float32.
        - 1.0 if sum(agent_states) == goal_sum.
        - 0.0 otherwise. (Team reward)

    - Termination: Episode ends if the goal is achieved.
    - Truncation: Episode ends if `max_steps_in_episode` is reached.
    """

    def __init__(self, num_agents: int, time_limit: int):
        """
        Initializes the SwitchGameEnv.

        Args:
            num_agents: The number of agents in the environment. Must be >= 1.
            max_steps_in_episode: The maximum number of steps before an episode is truncated.
                                   Must be >= 1.
        """
        if not isinstance(num_agents, int) or num_agents < 1:
            raise ValueError(f"num_agents must be a positive integer, got {num_agents}.")
        if not isinstance(time_limit, int) or time_limit < 1:
            raise ValueError(f"max_steps_in_episode must be a positive integer, got {time_limit}.")

        self.num_agents = num_agents
        self.time_limit = time_limit
        self.action_dim = 1

        # Call super().__init__() to ensure specs are initialized via their cached_properties.
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"SwitchGameEnv(num_agents={self.num_agents}, max_steps_in_episode={self.time_limit})"
        )

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Defines the specification of the observation returned by the environment."""
        av = specs.Array(
            shape=(self.num_agents, 3),  # (num_agents, [my_state, sum_norm, goal_norm])
            dtype=jnp.float32,
            name="observation",
        )

        action_mask = specs.BoundedArray(
            shape=(self.num_agents, 2), dtype=bool, minimum=False, maximum=True, name="action_mask"
        )
        step_count = specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.time_limit,
            name="step_count",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=av,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Defines the specification of the actions that can be taken by the agents."""
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=jnp.int32,  # Actions are 0 or 1
            minimum=0,
            maximum=1,
            name="action",
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[SwitchGameState, TimeStep[chex.Array]]:
        """Resets the environment to an initial state."""
        key, states_key, goal_key, new_state_key = jax.random.split(key, 4)

        initial_agent_states = jax.random.randint(
            states_key,
            shape=(self.num_agents,),
            minval=0,
            maxval=2,  # Exclusive upper bound for randint, so generates 0 or 1
            dtype=jnp.int32,
        )
        initial_goal_sum = jax.random.randint(
            goal_key,
            shape=(),  # Scalar for the sum
            minval=1,
            maxval=self.num_agents,  # Inclusive range [1, num_agents-1]
            dtype=jnp.int32,
        )

        # if you randomly init solved then change agent 0
        # initial_agent_states = jax.lax.cond(
        #     jnp.sum(initial_agent_states) == initial_goal_sum,
        #     lambda x: x.at[0].set(1 - x[0]),
        #     lambda x: x,
        #     initial_agent_states,
        # )

        initial_observations = self._make_all_observations(
            initial_agent_states,
            initial_goal_sum,
            jnp.zeros((), dtype=jnp.int32),
        )

        # Create the first TimeStep using jumanji.types.restart
        # timestep = restart(observation=initial_observations)
        # timestep.extras = {"env_metrics": {}}

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.zeros((self.num_agents,), dtype=jnp.float32),
            discount=jnp.ones((), dtype=jnp.float32),
            observation=initial_observations,
            extras={"env_metrics": {"won_episode": False}},
        )

        state = SwitchGameState(
            key=new_state_key,  # Store a new key for the next step if needed
            agent_int_states=initial_agent_states,
            goal_sum=initial_goal_sum,
            step_count=jnp.int32(0),
        )
        return state, timestep

    def step(
        self, state: SwitchGameState, action: chex.Array
    ) -> Tuple[SwitchGameState, TimeStep[chex.Array]]:
        """Run one timestep of the environment's dynamics."""
        # Validate action, or assume it conforms to spec. For JAX, inputs are usually trusted.
        # action is expected to be of shape (num_agents,) and dtype int32 with values 0 or 1.

        # Update agent states: new_state = (old_state + action) % 2
        # If action is 0, state remains. If action is 1, state flips (0->1, 1->0).
        action = action.squeeze(axis=1)
        # new_agent_int_states = (state.agent_int_states + action.astype(jnp.int32)) % 2
        new_agent_int_states = action.astype(jnp.int32)

        # Calculate sum and determine if the goal is achieved
        current_sum_of_states = jnp.sum(new_agent_int_states)
        goal_achieved = current_sum_of_states == state.goal_sum

        # Update step count and check for time limit
        new_step_count = state.step_count + 1
        time_limit_reached = new_step_count >= self.time_limit

        # Create observations for the new state. Goal sum is constant for the episode.
        observations = self._make_all_observations(
            new_agent_int_states, state.goal_sum, state.step_count
        )

        # Determine TimeStep type (termination, truncation, or transition)
        # These functions are defined locally to capture `observations`.
        # They are argument-less as jax.lax.cond will call them without arguments
        # if no operands are passed to cond itself.
        def _goal_achieved_branch_fn():
            return termination(
                reward=jnp.ones((self.num_agents,), jnp.float32), observation=observations
            )

        def _not_goal_achieved_branch_fn():
            # This branch is taken if goal_achieved is false.
            # Now check for time limit.
            rew = -jnp.abs(state.goal_sum - current_sum_of_states)

            def _time_limit_reached_branch_fn():
                return termination(
                    # reward=jnp.zeros((self.num_agents,), jnp.float32),
                    reward=jnp.full((self.num_agents,), rew, jnp.float32),
                    observation=observations,
                )

            def _continue_episode_branch_fn():
                # Reward is 0.0 as goal was not achieved and not time limit.
                return transition(
                    reward=jnp.full((self.num_agents,), rew, jnp.float32),
                    observation=observations,
                )

            return jax.lax.cond(
                time_limit_reached,
                _time_limit_reached_branch_fn,
                _continue_episode_branch_fn,
            )

        timestep = jax.lax.cond(
            goal_achieved,
            _goal_achieved_branch_fn,
            _not_goal_achieved_branch_fn,
        )
        timestep.extras = {"env_metrics": {"won_episode": goal_achieved}}

        # Update the environment state
        new_key, _ = jax.random.split(state.key)  # Get a new key for the next state
        new_env_state = SwitchGameState(
            key=new_key,
            agent_int_states=new_agent_int_states,
            goal_sum=state.goal_sum,  # Goal sum is constant throughout an episode
            step_count=new_step_count,
        )

        return new_env_state, timestep

    def _make_all_observations(
        self, agent_int_states: chex.Array, goal_sum: jnp.int32, step_count: int
    ) -> chex.Array:
        """
        Creates the observation array for all agents.
        Output shape: (num_agents, 3)
        """
        agent_indices = jnp.arange(self.num_agents)
        current_val = jnp.sum(agent_int_states)
        # Define a helper function to be vmapped.
        # It calculates observation for one agent given its index and global info.
        # def _get_obs_for_single_agent(
        #     idx: int, all_states: chex.Array, current_goal_sum: jnp.int32, num_agents: int
        # ):
        #     my_state_val_float = all_states[idx].astype(jnp.float32)
        #
        #     sum_all_states_float = jnp.sum(all_states).astype(jnp.float32)
        #     # num_agents is passed as an argument to be a clear dependency for JAX.
        #     sum_all_states_normalized = sum_all_states_float / num_agents
        #
        #     goal_sum_float = current_goal_sum.astype(jnp.float32)
        #     goal_sum_normalized = goal_sum_float / num_agents
        #
        #     return jnp.array(
        #         [my_state_val_float, sum_all_states_normalized, goal_sum_normalized],
        #         dtype=jnp.float32,
        #     )

        def get_obs(my_state: chex.Array):
            my_state_val_float = my_state.astype(jnp.float32)

            current_normalised = current_val.astype(jnp.float32) / self.num_agents
            goal_normalized = goal_sum.astype(jnp.float32) / self.num_agents

            # print(
            #     f"my: {my_state_val_float.shape} | sum: {sum_all_states_normalized.shape} | goal: {goal_sum_normalized.shape}"
            # )
            # jax.debug.print(
            #     "my_state_val {m} | sum: {s} | goal: {g}",
            #     m=my_state_val_float,
            #     s=sum_all_states_normalized,
            #     g=goal_sum_normalized,
            # )
            return jnp.array(
                [my_state_val_float, current_normalised, goal_normalized],
                dtype=jnp.float32,
            )

        # Vmap the helper function.
        # in_axes: (index_to_map_over, broadcast, broadcast, static_broadcast)
        # self._num_agents is treated as a static argument for vmap here.
        # all_observations = jax.vmap(_get_obs_for_single_agent, in_axes=(0, None, None, None))(
        #     agent_indices, agent_int_states, goal_sum, self.num_agents
        # )
        all_observations = jax.vmap(get_obs)(agent_int_states)

        obs = Observation(
            agents_view=all_observations,
            action_mask=jnp.full((self.num_agents, 2), True),
            step_count=jnp.full((self.num_agents,), step_count, dtype=jnp.int32),
        )

        return obs
