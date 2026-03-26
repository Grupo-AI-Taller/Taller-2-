from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        # TODO: Implement your code here
        return None


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        alpha, beta = float("-inf"), float("inf")
        best_value = float("-inf")
        best_action = None

        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            value = self.min_value(successor, 1, 0, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

    def max_value(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        """
        Nodo MAX (dron).
        """
        if state.is_terminal() or depth == self.depth:
            return self.evaluation_function(state)

        value = float("-inf")
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            child_value = self.min_value(successor, 1, depth, alpha, beta)
            if child_value > value:
                value = child_value
            if value > beta: 
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, state: GameState, agent_index: int, depth: int, alpha: float, beta: float) -> float:
        """
        Nodo MIN (cazadores).
        """
        if state.is_terminal():
            return self.evaluation_function(state)

        value = float("inf")
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)

            if agent_index < state.get_num_agents() - 1:
                value = min(value, self.min_value(successor, agent_index + 1, depth, alpha, beta))
            else:
                value = min(value, self.max_value(successor, depth + 1, alpha, beta))

            if value < alpha: 
                return value
            beta = min(beta, value)
        return value



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

         Versión inicial:
        
         def get_action(self, state):
             num_agents = state.get_num_agents()
        
             def expectimax(state, agent_index, depth):
                 if state.is_win() or state.is_lose() or depth == 0:
                     return self.evaluation_function(state)
        
                 legal_actions = state.get_legal_actions(agent_index)
                 next_agent = (agent_index + 1) % num_agents
        
                 if agent_index == 0:  # MAX
                     best_val = float('-inf')
                     for action in legal_actions:
                         val = expectimax(state.generate_successor(agent_index, action), next_agent, depth - 1)
                         if val > best_val:
                             best_val = val
                     return best_val
                 else:  # promedio
                     total = 0
                     for action in legal_actions:
                         total += expectimax(state.generate_successor(agent_index, action), next_agent, depth - 1)
                     return total / len(legal_actions)
        
             legal_actions = state.get_legal_actions(self.index)
             best_val = float('-inf')
             best_action = legal_actions[0]  # Error: no verifica si la lista está vacía
             for action in legal_actions:
                 val = expectimax(state.generate_successor(self.index, action), 1, self.depth - 1)
                 if val > best_val:
                     best_val = val
                     best_action = action
             return best_action
        
         Prompt usado para mejorar:
         "Ayúdame a mejorar esta implementación del ExpectimaxAgent. Los cazadores siguen una política
         mixta, con probabilidad p actúan al azar y con probabilidad (1-p) actúan
         greedy. La profundidad debe decrementarse solo cuando todos los agentes
         hayan movido (turno completo)."
        """
        num_agents = state.get_num_agents()

        def expectimax(state: GameState, agent_index: int, depth: int) -> float:
            # Terminal o límite de profundidad
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            if agent_index == 0:
                # Nodo MAX (dron)
                return max(
                    expectimax(state.generate_successor(agent_index, action), next_agent, next_depth)
                    for action in legal_actions
                )
            else:
                # Nodo de AZAR (cazador) con modelo mixto
                child_values = [
                    expectimax(state.generate_successor(agent_index, action), next_agent, next_depth)
                    for action in legal_actions
                ]
                p = self.prob
                min_val = min(child_values)
                mean_val = sum(child_values) / len(child_values)
                return (1 - p) * min_val + p * mean_val

        # Elegir la mejor acción para el dron (agente 0)
        legal_actions = state.get_legal_actions(self.index)
        if not legal_actions:
            return None

        best_action = None
        best_value = float('-inf')
        for action in legal_actions:
            successor = state.generate_successor(self.index, action)
            next_agent = (self.index + 1) % num_agents
            value = expectimax(successor, next_agent, self.depth)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
