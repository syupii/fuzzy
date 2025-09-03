from typing import Dict, Any, Optional
import uuid
import numpy as np

class GeneticIndividual:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.genome: np.ndarray = None
        self.fitness: float = 0.0
        self.age: int = 0
        self.tree_structure: Optional[Dict[str, Any]] = None
    
    def initialize_random(self, genome_length: int):
        self.genome = np.random.random(genome_length)
    
    def mutate(self, mutation_rate: float):
        mask = np.random.random(len(self.genome)) < mutation_rate
        self.genome[mask] += np.random.normal(0, 0.1, np.sum(mask))
        self.genome = np.clip(self.genome, 0, 1)