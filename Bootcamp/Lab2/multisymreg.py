import random
import operator

import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Set a seed for randomization so that every trial yields the same result
random.seed(25)

# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1, -1))  # Set multiple objectives
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)  # represented as tree structure

# Initialize a primitive set that contains all the primitives we can use
pset = gp.PrimitiveSet("Main", arity=1)  # arity = amount of arguments each primitive takes
pset.addPrimitive(np.add, arity=2)
pset.addPrimitive(np.subtract, arity=2)
pset.addPrimitive(np.multiply, arity=2)
pset.addPrimitive(np.negative, arity=1)
pset.addPrimitive(np.sin, arity=1)
pset.addPrimitive(np.cos, arity=1)
pset.addPrimitive(np.tan, arity=1)
pset.renameArguments(ARG0="x")

# Define toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval_symb_reg(individual, points, pset):
    # Compile the tree into a function
    func = gp.compile(expr=individual, pset=pset)
    # Calculate main squared error between this function and the target function
    sq_errors = (func(points) - (np.negative(points) + np.sin(points ** 2) + np.tan(points ** 3) + np.cos(points))) ** 2
    # Set standard dev and length of individual as objectives
    return np.sqrt(np.sum(sq_errors) / len(points)), len(individual)


# Register genetic operators
toolbox.register("evaluate", eval_symb_reg, points=np.linspace(-1, 1, 1000), pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Add tree height constraints to crossover and mutation functions
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# Returns true if the first individual dominates the second individual
def pareto_dominance(ind1, ind2):
    not_equal = False
    for value1, value2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value1 > value2:
            return False
        elif value1 < value2:
            not_equal = True
    return not_equal


# Initialize a random population of 300
pop = toolbox.population(n=300)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Initialize a separate individual for comparison
a_given_individual = toolbox.population(n=1)[0]
a_given_individual.fitness.values = toolbox.evaluate(a_given_individual)

# Sort the population by pareto dominance in comparison to the separate individual
dominated = [ind for ind in pop if pareto_dominance(a_given_individual, ind)]
dominators = [ind for ind in pop if pareto_dominance(ind, a_given_individual)]
others = [ind for ind in pop if not ind in dominated and not ind in dominators]

# Plot the objective space using sorted population
for ind in dominators:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', alpha=0.7)
for ind in dominated:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'g.', alpha=0.7)
for ind in others:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'k.', alpha=0.7, ms=3)
plt.plot(a_given_individual.fitness.values[0], a_given_individual.fitness.values[1], 'bo', ms=6)
plt.xlabel('Mean Squared Error')
plt.ylabel('Tree Size')
plt.title('Objective space')
plt.tight_layout()
plt.show()

# Main evolutionary algorithm
NGEN = 50
MU = 50
LAMBDA = 100
CXPB = 0.5
MUTPB = 0.2

pop = toolbox.population(n=MU)
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

# Plot the result of out run and display the best individual
print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
plt.plot(gen, avg, label="average")
plt.plot(gen, min_, label="minimum")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="upper left")
plt.show()

# Split fitness values into separate lists
fitness_1 = [ind.fitness.values[0] for ind in hof]
fitness_2 = [ind.fitness.values[1] for ind in hof]
pop_1 = [ind.fitness.values[0] for ind in pop]
pop_2 = [ind.fitness.values[1] for ind in pop]

# Print dominated population for debugging
# for ind in pop:
#     print(ind.fitness)

plt.scatter(pop_1, pop_2, color='b')
plt.scatter(fitness_1, fitness_2, color='r')
plt.plot(fitness_1, fitness_2, color='r', drawstyle='steps-post')
plt.xlabel("Mean Squared Error")
plt.ylabel("Tree Size")
plt.title("Pareto Front")
plt.show()

f1 = np.array(fitness_1)
f2 = np.array(fitness_2)

# Calculate area under curve with least squares method
print("Area Under Curve: %s" % (np.sum(np.abs(np.diff(f1))*f2[:-1])))
