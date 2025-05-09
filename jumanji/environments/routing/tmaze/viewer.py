from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle, Circle, Polygon


class TmazeViewer:
    def __init__(
        self, name: str = "TMaze", render_mode: str = "human", length: int = 5, width: int = 2
    ) -> None:
        """
        Viewer for a `TMaze` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self.length = length
        self.width = width
        self.height = width
        self.top_target_x = length + width

        self.colors = {
            -1: (0.5, 0.5, 0.5, 1.0),  # Out of bounds/empty: Grey
            0: (1.0, 1.0, 1.0, 1.0),  # Empty cell: White
            1: (0.0, 0.0, 1.0, 1.0),  # Agent 1: Blue
            2: (1.0, 0.0, 0.0, 1.0),  # Agent 2: Red
        }

        self.target_colors = {
            0: (0.0, 1.0, 0.0, 1.0),  # Target for agent with target index 0: Green
            1: (1.0, 0.5, 0.0, 1.0),  # Target for agent with target index 1: Orange
            2: (0.5, 0.0, 1.0, 1.0),  # Purple for shared target
        }

        self.fig, self.ax = plt.subplots()
        self.render_mode = render_mode
        if render_mode == "human":
            plt.ion()  # Turn on interactive mode for live updates

    def render(self, state, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Render TMaze.

        Args:
            state: the state of the TMaze environment to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self.ax.clear()
        self._draw_grid(state, self.ax)
        self.ax.set_axis_off()
        self.ax.set_aspect(1)
        self.ax.relim()
        self.ax.autoscale_view()

        if self.render_mode == "human":
            plt.pause(0.001)  # Small pause for interactive rendering
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.array(self.fig.canvas.renderer.buffer_rgba())

        if save_path:
            self.fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        return None

    def _draw_grid(self, state, ax: plt.Axes) -> None:
        """Draws the TMaze grid, including the new corridor."""

        # Draw the main T part and the new corridor
        for x in range(self.length + self.width + 1):  # Extend range for corridor
            for y in range(-self.width, self.width + 2):
                if not self._is_cell_in_bounds(jnp.array([x, y])):
                    continue
                self._draw_grid_cell(x, y, ax)

        # Draw the agents
        self._draw_agent(state.agent_positions[0], 1, ax)
        self._draw_agent(state.agent_positions[1], 2, ax)

        # Draw the targets, handling the same_target case
        if state.same_target:
            self._draw_shared_target(jnp.array([self.top_target_x + 0.5, 0.5]), ax)
            self._draw_shared_target(jnp.array([self.top_target_x + 0.5, 1.5]), ax)
        else:
            self._draw_target(state.agent_targets[0], 0, ax)
            self._draw_target(state.agent_targets[1], 1, ax)

    def _is_cell_in_bounds(self, cell_pos: jax.Array) -> bool:
        """Checks if a cell is within the bounds of the TMaze, including the extended corridor."""
        x, y = cell_pos
        is_on_vertical = (x >= 0) & (x <= self.length + self.width)  # Extend range for corridor
        is_on_horizontal = x == self.length

        return (is_on_vertical & ((y == 0) | (y == 1))) | (
            is_on_horizontal & (y >= -self.width) & (y <= 1 + self.width)
        )

    def _draw_grid_cell(self, col: int, row: int, ax: plt.Axes) -> None:
        """Draws a single grid cell."""
        cell_value = 0  # Default to empty

        cell = Rectangle(
            (col, row), 1, 1, facecolor=self.colors[cell_value], edgecolor="black", linewidth=1
        )
        ax.add_patch(cell)

    def _draw_agent(self, position: jax.Array, agent_id: int, ax: plt.Axes) -> None:
        """Draws an agent as a circle."""
        x, y = position
        circle = Circle((x + 0.5, y + 0.5), 0.4, color=self.colors[agent_id])
        ax.add_patch(circle)

    def _draw_target(self, position: jax.Array, target_id: int, ax: plt.Axes) -> None:
        """Draws a target as a triangle."""
        x, y = position
        triangle = Polygon(
            [[x + 0.5, y + 0.8], [x + 0.2, y + 0.2], [x + 0.8, y + 0.2]],
            color=self.target_colors[target_id],
        )
        ax.add_patch(triangle)

    def _draw_shared_target(self, position: jax.Array, ax: plt.Axes) -> None:
        """Draws the shared target at the end of the corridor."""
        x, y = position
        diamond = Polygon(
            [[x, y + 0.4], [x + 0.4, y], [x, y - 0.4], [x - 0.4, y]],
            color=self.target_colors[2],  # Use purple for shared target
        )
        ax.add_patch(diamond)

    def animate(
        self,
        states: Sequence[chex.Array],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of TMaze grids.

        Args:
            grids: sequence of TMaze grids corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self.fig, self.ax  # Reuse existing figure and axes
        plt.close(fig=fig)

        def make_frame(state) -> Tuple[Artist]:
            ax.clear()
            self._draw_grid(state, ax)
            ax.set_axis_off()
            ax.set_aspect(1)
            ax.relim()
            ax.autoscale_view()
            return (ax,)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation
