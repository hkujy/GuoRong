

from cmath import log
import imp
from math import fabs
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
import matplotlib.pyplot as plt
import para
import os
import funs
import random
from deap import algorithms,base,creator,tools
import math

def decode(chrom,nbit,nvar,lb,ub):
    """
        convert binary chrom to float variables
        nbit: number of bits for a one decision variable
        nvar: number of variables
        chrom: invidual chromsome 
        nbit*nvar = length of chromsome  
        lb: lower bound of the decsion variables
        ub: ub bound of the decision variables
    """
    # print("Binary Rep = ", end='')
    # for i in chrom:
        # print ('{0},'.format(i),end='')
    # print()
    quant = []
    for i in range(0,nbit):
        quant.append(math.pow(0.5,i+1))
    sum_quant = sum(quant)
    for i in range(0, len(quant)):
        quant[i] = quant[i]/sum_quant
    # print ('quant = ',quant)
    float_var = [] # decoded float decision variables
    for i in range(0, nvar):
        val = 0
        for j in range(0, nbit):
            val += quant[j]*chrom[i*nbit+j]            
        float_var.append(val*(ub[i]-lb[i])+lb[i])

    # print("float val = ", end='')
    # for i in float_var:
        # print ('{0},'.format(i),end='')
    # print()
    # os.system('pause')
     
    return float_var


def get_fitness(individual):
    # Step 1: decode binary representation to float decision variables
    fre = decode(individual,para.nbit_per_var,para.nvars,para.lb,para.ub)
    fitness_val = evaluate(fre)
    return fitness_val,


def ga():
    #https://github.com/hkujy/PT42185/blob/master/PT2019/trails/ga.py
 
    crossover_rate = 0.5
    mutation_rate = 0.2

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Structure initializers
    # define how to init a individual 
    # n= 3 means there are four transit lines
    toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, n=para.nvars*para.nbit_per_var)
    # based on the init for individual, initialize a population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", get_fitness)
    # define cross over and mutation operation
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    random.seed(64)
    # step 1: creat the population
    pop = toolbox.population(n=para.npop)  
    # CXPB, MUTPB = 0.5, 0.2
    # step 2: evaluate the population
    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit  in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    # Evolv the population
    g = 0
    # while max(fits) < 100 and g < 10:
    while g < para.ngen:
        g = g + 1
        print(" --- Generation %i ----" % g)
        # select the net generation individual
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # next is to apply crossover and mutaktion on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring    
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        var = decode(best_ind,para.nbit_per_var,para.nvars,para.lb,para.ub)
        print(var)
