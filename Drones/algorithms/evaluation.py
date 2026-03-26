from __future__ import annotations

from typing import TYPE_CHECKING
from math import inf
from algorithms.utils import bfs_distance


if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    # TODO: Implement your code here
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    layout = state.get_layout()
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = state.get_pending_deliveries()
    current_score = state.get_score()

    if not pending_deliveries:
        return min(1000.0, current_score + 500.0)

    delivery_distances = []
    for delivery in pending_deliveries:
        dist = bfs_distance(layout, drone_pos, delivery, hunter_restricted=False)
        if dist != inf:
            delivery_distances.append(dist)

    if delivery_distances:
        nearest_delivery_dist = min(delivery_distances)
    else:
        nearest_delivery_dist = 999

    hunter_distances = []
    for hunter in hunter_positions:
        dist = bfs_distance(layout, hunter, drone_pos, hunter_restricted=True)
        hunter_distances.append(dist)

    if hunter_distances:
        nearest_hunter_dist = min(hunter_distances)
    else:
        nearest_hunter_dist = inf

    num_pending = len(pending_deliveries)

    urgency_bonus = 0.0
    for delivery in pending_deliveries:
        drone_to_delivery = bfs_distance(layout, drone_pos, delivery, hunter_restricted=False)
        if drone_to_delivery == inf:
            continue

        hunter_to_delivery = inf
        for hunter in hunter_positions:
            d = bfs_distance(layout, hunter, delivery, hunter_restricted=True)
            hunter_to_delivery = min(hunter_to_delivery, d)

        if drone_to_delivery < hunter_to_delivery:
            urgency_bonus = max(urgency_bonus, 25.0 / (1.0 + drone_to_delivery))

    value = 0.0
    value += 1.5 * current_score
    value -= 40.0 * num_pending
    value -= 6.0 * nearest_delivery_dist

    if nearest_hunter_dist == inf:
        value += 80.0
    else:
        value += 6.0 * nearest_hunter_dist

    value += urgency_bonus

    if nearest_hunter_dist == 0:
        return -1000.0
    elif nearest_hunter_dist == 1:
        value -= 400.0
    elif nearest_hunter_dist == 2:
        value -= 120.0

    value = max(-1000.0, min(1000.0, value))
    return value
