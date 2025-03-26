import torch

from typing import Literal, Tuple


from einops import rearrange

from gfn.containers import Trajectories
from gfn.actions import Actions
from gfn.states import States
from gfn.env import Env, DiscreteEnv
from gfn.gym.helpers.preprocessors import KHotPreprocessor, OneHotPreprocessor
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from gfn.states import DiscreteStates



class GenomeAssemblyEnv(DiscreteEnv):
    def __init__(self, kmer_length: int, dataset: torch.Tensor, height: int = 4,
                 device_str: Literal["cpu", "cuda"] = "cpu",
                 preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"] = "KHot"):
        # Initialize environment parameters specific to genome assembly
        self.kmer_length = kmer_length
        self.dataset = dataset  # This could be the PhiX dataset or any other genomic data

        # Define preprocessor based on chosen encoding scheme (KHot, OneHot, etc.)
        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=height, ndim=kmer_length)
        elif preprocessor_name == "OneHot":
            preprocessor = OneHotPreprocessor(n_states=self.n_states, get_states_indices=self.get_states_indices)
        else:
            preprocessor = IdentityPreprocessor(output_dim=kmer_length)

        # Define other initialization parameters like actions and rewards
        n_actions = self.kmer_length + 1  # Example: Actions could be extending the assembly
        s0 = torch.zeros(kmer_length, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((kmer_length,), fill_value=-1, dtype=torch.long, device=torch.device(device_str))

        # Call the parent class's constructor
        super().__init__(n_actions=n_actions, s0=s0, state_shape=(kmer_length,), sf=sf, device_str=device_str,
                         preprocessor=preprocessor)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """Define a reward function based on assembly quality."""
        # For example, align the final assembly with the reference dataset and calculate a similarity score
        final_assembly = final_states.tensor  # Final sequence
        similarity_score = self.calculate_similarity(final_assembly)
        return similarity_score

    def calculate_similarity(self, final_assembly: torch.Tensor) -> torch.Tensor:
        """Compute similarity (e.g., using sequence alignment) between the assembly and the target dataset."""
        # Placeholder for sequence alignment algorithm (e.g., Needleman-Wunsch)
        # Return a similarity score as the reward
        return torch.rand(final_assembly.shape[0], device=self.device)  # Dummy reward for now

    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        """Define how a step modifies the current assembly state."""
        # Apply the action (e.g., extend the assembly) and return the new state
        return self.apply_action(states.tensor, actions.tensor)

    def apply_action(self, current_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply action to extend or modify the assembly state."""
        # Example of applying an action to extend the current assembly
        new_state = current_state  # Modify based on action (e.g., appending a k-mer)
        return new_state
