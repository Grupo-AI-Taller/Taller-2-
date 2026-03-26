from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    # TODO: Implement your code here
    return None


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    
    return None

def forward_check(csp: DroneAssignmentCSP, var: str, value: str, assignment: dict[str, str]) -> tuple[bool, dict[str, list[str]]]:
    """
    Aplica forward checking después de asignar un valor a una variable.
    Retorna (éxito, dominios_guardados).
    - éxito = False si algún vecino queda sin valores posibles.
    - dominios_guardados = copia de los dominios antes de modificarlos (para restaurar).
    """
    saved_domains = {v: list(csp.domains[v]) for v in csp.domains}

    for neighbor in csp.get_neighbors(var):
        if neighbor not in assignment:
            new_domain = []
            for val in csp.domains[neighbor]:
                if csp.is_consistent(neighbor, val, assignment):
                    new_domain.append(val)
            csp.domains[neighbor] = new_domain
            if len(new_domain) == 0:
                return False, saved_domains

    return True, saved_domains

def backtrack_fc(csp: DroneAssignmentCSP, assignment: dict[str, str], stats: dict[str, int]) -> dict[str, str] | None:
    """
    Función recursiva de backtracking con forward checking.
    """
    if csp.is_complete(assignment):
        return assignment

    # Seleccionamos la siguiente variable no asignada
    unassigned = csp.get_unassigned_variables(assignment)
    var = unassigned[0]

    for value in csp.domains[var]:
        if csp.is_consistent(var, value, assignment):
            # Asignamos valor
            csp.assign(var, value, assignment)
            stats["assignments"] += 1

            # Forward checking
            success, saved_domains = forward_check(csp, var, value, assignment)
            if success:
                result = backtrack_fc(csp, assignment, stats)
                if result is not None:
                    return result

            # Restauramos dominios y desasignamos
            csp.domains = {v: list(saved_domains[v]) for v in saved_domains}
            csp.unassign(var, assignment)

    stats["backtracks"] += 1
    return None

def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    stats = {"assignments": 0, "backtracks": 0}
    solution = backtrack_fc(csp, {}, stats)
    print(f"[Forward Checking] Assignments: {stats['assignments']}, Backtracks: {stats['backtracks']}")
    return solution


def revise(csp: DroneAssignmentCSP, domains: dict[str, list[str]], xi: str, xj: str) -> bool:
    revised = False
    to_remove = []

    for vi in domains[xi]:
        supported = False
        for vj in domains[xj]:
            temp_assignment = {xi: vi}
            if csp.is_consistent(xj, vj, temp_assignment):
                supported = True
                break
        if not supported:
            to_remove.append(vi)

    for vi in to_remove:
        domains[xi].remove(vi)
        revised = True

    return revised


def ac3(csp: DroneAssignmentCSP, domains: dict[str, list[str]], queue=None) -> bool:
    if queue is None:
        queue = deque()
        for xi in domains:
            for xj in csp.get_neighbors(xi):
                queue.append((xi, xj))

    while queue:
        xi, xj = queue.popleft()

        if revise(csp, domains, xi, xj):
            if len(domains[xi]) == 0:
                return False

            for xk in csp.get_neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))

    return True


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    domains = deepcopy(csp.domains)

    if not ac3(csp, domains):
        return None

    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return assignment.copy()

        var = csp.get_unassigned_variables(assignment)[0]

        for value in list(domains[var]):
            if csp.is_consistent(var, value, assignment):
                saved_domains = deepcopy(domains)

                csp.assign(var, value, assignment)
                domains[var] = [value]

                queue = deque((neighbor, var) for neighbor in csp.get_neighbors(var))

                if ac3(csp, domains, queue):
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                csp.unassign(var, assignment)
                domains.clear()
                domains.update(saved_domains)

        return None

    return backtrack({})


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    # TODO: Implement your code here (BONUS)
    return None
