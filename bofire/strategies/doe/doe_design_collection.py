from typing import List, Optional
import pandas as pd
from bofire.strategies.doe.objective import Objective
import numpy as np

#TODO: test

class DoEDesignCollection:

    def __init__(self, designs: List[pd.DataFrame], optimality_criterion: Optional[Objective] = None):
        """A class to store a collection of designs of a common size as they are produced when running 
        a DoE strategy multiple times with the same settings.

        Args:
            designs (List[pd.DataFrame]): A list of designs.
            optimality_criterion (Objective, optional): The optimality criterion used to evaluate the designs. Defaults to None.
        """
        assert len(designs) > 0
        for design in designs:
            assert design.shape[0] == designs[0].shape[0]
            assert design.shape[1] == designs[0].shape[1]

        self.designs = designs
        self.optimality_criterion = optimality_criterion

    def __len__(self) -> int:
        """Returns the number of designs in the collection."""
        return len(self.designs)
    
    def n_experiments(self) -> int:
        """Returns the number of experiments in each design."""
        return self.designs.shape[0]
    
    def n_variables(self) -> int:
        """Returns the number of variables in each design."""
        return self.designs.shape[1]

    def champion(self) -> pd.DataFrame:
        """Returns the champion design of the collection. The best design w.r.t. the optimality criterion."""
        if self.optimality_criterion is None:
            raise ValueError("No optimality criterion specified.")
        optimality = [self.optimality_criterion.evaluate(design.to_numpy().flatten()) for design in self.designs]
        return self.designs[np.argmin(optimality)]

    def evaluate(self):
        """Evaluates the optimality criterion for each design in the collection.
        
        Returns:
            np.array: An array of the optimality criterion values.
        """
        if self.optimality_criterion is None:
            raise ValueError("No optimality criterion specified.")
        return np.array([self.optimality_criterion.evaluate(design.to_numpy.flatten()) for design in self.designs])