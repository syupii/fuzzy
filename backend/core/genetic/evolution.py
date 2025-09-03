from typing import List, Callable
from .individual import GeneticIndividual
from .population import Population
from .operators import GeneticOperators

class EvolutionEngine:
    def __init__(self, config):
        self.config = config
        self.operators = GeneticOperators(config)
        self.generation = 0
    
    def evolve(self, population: Population, 
               fitness_function: Callable) -> Population:
        # 適応度評価
        for individual in population.individuals:
            individual.fitness = fitness_function(individual)
        
        # 選択・交叉・突然変異
        new_population = self.operators.evolve_generation(population)
        self.generation += 1
        
        return new_population