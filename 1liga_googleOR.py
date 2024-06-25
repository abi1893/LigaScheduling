
from model_googleOR import intermediateSolutionHandler
import pandas as pd
from ortools.sat.python import cp_model
import numpy as np
import util as u
from datetime import datetime
import os

from model_googleOR import intermediateSolutionHandler
import pandas as pd
from ortools.sat.python import cp_model
import numpy as np
import util as u
from datetime import datetime
import os


def create_model(df, groupsize=4, matchdays=1, forceHost=True):
    model = cp_model.CpModel()
    teams = range(len(df))
    days = range(matchdays)
    pairs = [(i, j) for i in teams for j in teams if i < j]
    M, C = {}, {}

    # Variable creation
    for k in days:
        for i in teams:
            for j in teams:
                M[(k, i, j)] = model.NewBoolVar(f"M_{k}_{i}_{j}")

    # Constraint creation
    # Each team plays exactly one game per day
    for k in days:
        for j in teams:
            model.AddExactlyOne(M[(k, i, j)] for i in teams)

    for k in days:
        for i in teams:
            # If team i hosts, exactly groupsize teams play at i
            model.Add(sum(M[(k, i, j)] for j in teams) == groupsize).OnlyEnforceIf(M[(k, i, i)])
            # If team i does not host, no one plays at i
            model.Add(sum(M[(k, i, j)] for j in teams) == 0).OnlyEnforceIf(M[(k, i, i)].Not())

    # No pair plays twice
    for k in days:
        for h in teams:
            for (i, j) in pairs:
                C[(k, h, i, j)] = model.NewBoolVar(f"C_{k}_{i}_{j}")
                model.Add(M[k, h, i] + M[k, h, j] == 2).OnlyEnforceIf(C[(k, h, i, j)])
                model.Add(M[k, h, i] + M[k, h, j] <= 1).OnlyEnforceIf(C[(k, h, i, j)].Not())
    for i, j in pairs:
        model.Add(sum(C[(k, h, i, j)] for k in days for h in teams) <= 2)

    if forceHost:
        # All teams host at least once
        for i in teams:
            model.Add(sum(M[(k, i, i)] for k in days) > 0)
            model.Add(sum(M[(k, i, i)] for k in days) <= 2)

    for k in days:
        for i in teams:
            for j in teams:
                if df.loc[i][str(j)] > 600:
                    model.Add(M[(k, i, j)] == 0)

    # Objective: minimize total distance traveled and ensure balanced travel distribution
    objective_terms = []
    team_distances = {i: [] for i in teams}

    for k in days:
        for i in teams:
            for j in teams:
                dist = df.loc[i][str(j)]
                term = M[(k, i, j)] * dist
                objective_terms.append(term)
                team_distances[i].append(term)

    model.Minimize(sum(objective_terms))



    return model, M


if __name__ == "__main__":
    file = "input/2buli_distMatrix.txt"
    n_days = 4
    n_teams = 44
    n_size = 4
    forceHost = False
    TIME = 3600
    LIMIT = 0
    TIME_PER_SOLUTION = 0
    write = True

    df = u.init_df(file=file, max=n_teams)

    model, M = create_model(df, groupsize=n_size, matchdays=n_days, forceHost=forceHost)
    solver = cp_model.CpSolver()
    if (TIME != 0):
        solver.parameters.max_time_in_seconds = TIME
    solver.parameters.enumerate_all_solutions = True
    solutionHandler = intermediateSolutionHandler(LIMIT, TIME_PER_SOLUTION)
    print('start solving')
    status = solver.Solve(model, solutionHandler)
    solutionHandler.clean()
    if not (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE):
        print(f"No solution. Elapsed time: {solver.WallTime()}sec")
    else:
        objectiveValue = solver.ObjectiveValue()
        bestBound = solver.BestObjectiveBound()
        print(f"Finished with objective value {objectiveValue} in time: {int(solver.WallTime())}s")
        print(f"best objective bound: {bestBound}")
        sol = np.zeros((n_days, n_teams, n_teams), dtype=int)
        for (k, i, j) in M:
            if solver.Value(M[(k, i, j)]) == 1:
                sol[k, i, j] = 1
        allGroups = [u.getGroups(sol[k]) for k in range(n_days)]

        name = datetime.now().strftime(f"%m.%d._%H.%M_{int(solver.WallTime())}s_{int(objectiveValue)}")
        path = os.path.join(os.getcwd(), name)
        headline = f"run for {n_teams} teams in groups of {n_size} on {n_days} days, forced hosting: {forceHost}\n"
        headline += f"Computation Time given: {TIME}s (used {int(solver.WallTime())}s), objective value: {objectiveValue}\n"
        if (bestBound != 0):
            headline += f"best bound: {bestBound}\n"
        summary = u.evaluate(allGroups, df)
        if (write):
            with open(path + ".txt", "w") as file:
                file.write(headline + summary)
            coordinates = df['coords'].values.tolist()
            u.drawScatter(allGroups, coordinates, path)
