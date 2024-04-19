from didgelab.evo.nuevolution import MutationOperator, Genome
import numpy as np
from typing import Dict, Tuple

class SimpleMutation(MutationOperator):

    def apply(self, genome : Genome, evolution_parameters : Dict) -> Tuple[Genome, Dict]:

        mr = evolution_parameters["mutation_rate"]
        if mr is None:
            mr = 0.5
        mp = evolution_parameters["gene_mutation_prob"]
        if mp is None:
            mp = 0.5

        mutation = np.random.uniform(low=-mr, high=mr, size=len(genome.genome))
        mutation *= (np.random.sample(size=len(mutation))<mp).astype(int)
        print(genome.genome.round(2))
        print(mutation.round(2))
        mutation = genome.genome + mutation
        print(mutation.round(2))
        mutation[mutation<0] = 0
        mutation[mutation>1] = 1
        print(mutation.round(2))


        new_genome = genome.clone()
        new_genome.genome = mutation
        return new_genome, self.describe(genome, new_genome)

np.random.seed(0)
g = Genome(n_genes=5)

m = SimpleMutation()
params = {"mutation_rate": 0.5, "gene_mutation_prob": 0.5}
new_genome, _ = m.apply(g, params)


#print(g.genome.round(decimals=2))
#print(new_genome.genome.round(decimals=2))
#print(np.abs(g.genome-new_genome.genome).round(decimals=2))