from collections import defaultdict
import time
import torch
from torch import Tensor
from torch import nn
from torch import optim
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from gfn.gflownet import GFlowNet, FMGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.states import States, DiscreteStates, stack_states


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class TensorDict:
    def __init__(self, default_factory=None):
        # Use a defaultdict with an optional default factory
        self.data = defaultdict(default_factory)
        self.default_factory = default_factory

    def _tensor_to_hashable(self, tensor):
        # Recursively convert a tensor to a hashable structure (nested tuples)
        if isinstance(tensor, torch.Tensor):
            return self._tensor_to_hashable(tensor.tolist())
        elif isinstance(tensor, list):
            return tuple(self._tensor_to_hashable(item) for item in tensor)
        else:
            return tensor  # Base case: numbers are already hashable

    def __setitem__(self, tensor, value):
        # Convert tensor to a hashable structure for storage
        key = self._tensor_to_hashable(tensor)
        self.data[key] = value

    def __getitem__(self, tensor):
        # Retrieve value based on hashable structure
        key = self._tensor_to_hashable(tensor)
        return self.data[key]

    def __contains__(self, tensor):
        # Check existence based on hashable structure
        key = self._tensor_to_hashable(tensor)
        return key in self.data

    def __str__(self):
        # Pretty-print the dictionary content as tuples and values
        pretty_dict = {
            str(key): value for key, value in self.data.items()
        }
        return str(pretty_dict)

    def __repr__(self):
        # Provide a developer-friendly representation
        return f"TensorDict({self.__str__()})"


def get_all_transition_log_probs(env: DiscreteStates, pf_estimator):
    """
        Args:
    Returns transition_log_probs, a Tensor list with length of env.n_actions. 
        transition_log_probs[i][j] indicates the log probability of taking action i at a State env.all_states[j], i in [0, n_actions-1]
    """
    all_states = env.all_states
    estimator_output = pf_estimator(all_states)
    dist = pf_estimator.to_probability_distribution(all_states, estimator_output)
    transition_log_probs = [None] * env.n_actions

    for i in range(env.n_actions):
        action = torch.Tensor([i])
        transition_log_probs[i] = dist.log_prob(action)
    return transition_log_probs


def compute_log_probability(env: DiscreteStates, gfn, state: DiscreteStates, memo, transition_log_probs):
    """
    Recursively computes the log of the sampling probability π_θ(s) for a given terminal state `state`
    in a GFlowNet `gfn` using torchgfn library.

    Args:
        gfn (GFlowNet): The GFlowNet model instance.
        state (States): The terminal state for which we want to compute log π_θ(s).
        memo (dict): A dictionary for memoization to store previously computed log probabilities.

    Returns:
        torch.Tensor: The log probability π_θ(s).
    """
    if len(state.tensor.shape) == 1:
        state = stack_states([state])
    # Check if the result is already computed and stored in memo
    if state.tensor in memo:
        return memo[state.tensor]

    # Base case: if the state is the initial state, log π_θ(s_initial) = 0
    if state.is_initial_state.all():
        log_prob = torch.tensor([0.0], requires_grad=False)
        memo[state.tensor] = log_prob
        return log_prob

    # Recursive case: compute log π_θ(s) from all parent states
    # Collect log-probabilities for each parent transition
    log_probs = []
    # to iterate each parent state and the corresponding action
    for i in range(env.n_actions - 1):
        action = env.actions_from_tensor(torch.Tensor([[i]]).to(torch.int64))
        env.update_masks(state)
        if env.is_action_valid(state, action, backward=True):
            # s'
            parent_state_tensor = env.backward_step(state, action)
            parent_state = env.states_from_tensor(parent_state_tensor)
            # parent_state = stack_states([parent_state])
            parent_state_idx = env.get_states_indices(parent_state)
            # logPF(s|s'): Forward transition probability in log form
            log_forward_prob = transition_log_probs[i][parent_state_idx]
            # log π_θ(s'): Recursively compute log π_θ(parent_state)
            log_parent_prob = compute_log_probability(env, gfn, parent_state, memo, transition_log_probs)
            # Compute the sum inside the exponent for this parent
            log_probs.append(log_forward_prob + log_parent_prob)
    # Sum of exponentiated log-probabilities (log-sum-exp trick for numerical stability)
    log_prob = torch.logsumexp(torch.stack(log_probs), dim=0)
    # Memoize and return
    memo[state.tensor] = log_prob
    return log_prob


def compute_log_prob_termination(env: DiscreteStates, terminal_state: DiscreteStates, memo, transition_log_probs):
    if len(terminal_state.tensor.shape) == 1:
        terminal_state = stack_states([terminal_state])
    terminal_state_tensor = terminal_state.tensor
    termination_action = env.actions_from_tensor(torch.Tensor([[env.n_actions - 1]]).to(torch.int64))
    env.update_masks(terminal_state)
    assert env.is_action_valid(terminal_state, termination_action,
                               backward=False), f"Error: Termination at given state {terminal_state.tensor} is invalid!"
    terminal_state_idx = env.get_states_indices(terminal_state)
    # log π_θ(s_terminal) + log termination
    return memo[terminal_state_tensor] + transition_log_probs[-1][terminal_state_idx]


# to compute the sampling probability wit monte_carlo
def count_occurrences_with_monte_carlo(env: DiscreteStates, sampler, n_samples=10000, show_progress=True):
    """
    Computes the sampling probability of a given terminal state using Monte Carlo.

    Args:
        env: The environment instance.
        sampler: An initialized Sampler using the forward policy estimator.
        terminal_state: The terminal state whose probability we want to compute (as a tensor).
        n_samples: The number of trajectories to sample.

    Returns:
        occurrences: occurrences dict of each state.tensor
    """
    # Sample trajectories
    trajectories = sampler.sample_trajectories(env=env, n=n_samples)
    # Extract terminal states
    terminal_states = [traj.states[-2] for traj in trajectories]
    occurrences = TensorDict(int)
    for state in tqdm(terminal_states, desc="Processing terminal_states") if show_progress else terminal_states:
        occurrences[state.tensor] += 1
    return occurrences


def compute_log_prob_with_monte_carlo(occurrences, terminal_state, n_samples: int) -> float:
    # Calculate the probability
    if isinstance(terminal_state, States):
        if len(terminal_state.tensor.shape) == 1:
            terminal_state = stack_states([terminal_state])
        terminal_state = terminal_state.tensor
    return torch.log(torch.tensor(occurrences[terminal_state] / n_samples, requires_grad=False))


def get_random_test_set(env: HyperGrid, n=100):
    random_indices = torch.randperm(len(env.all_states))[:n]
    terminal_states = env.all_states[random_indices]
    log_rewards = torch.log(env.reward(terminal_states))
    return terminal_states, log_rewards


def get_sampled_test_set(gfn, env, n=100):
    # sampler = Sampler(estimator=gfn.pf)
    # test_trajectories = sampler.sample_trajectories(env=env, n=n)
    # terminal_states = test_trajectories.last_states
    terminal_states = gfn.sample_terminating_states(env=env, n=n)
    log_rewards = torch.log(env.reward(terminal_states))
    return terminal_states, log_rewards


@timer
def evaluate_GFNEvalS(gfn: GFlowNet, env: DiscreteStates, terminal_states, log_rewards, show_progress: bool = True):
    """Computes the GFNEvalS of given terminal states and log_rewards using Backtracking with memoization.

    Args:
        gfn: An initialized Sampler using the forward policy estimator.
        env: The HyperGrid environment instance.
        terminal_states: terminal states in test set.
        log_rewards: true rewards of terminal states in test set.

    Returns:
        spearman_corr_termination: Spearman's Rank Correlation (Modified GFNEvalS, including termination actions)
        memo: TensorDict, memo[s] indicates the probability from init_state to s, without counting the probability of termanating at state s.
        transition_log_probs: a Tensor list with length of env.n_actions. 
            transition_log_probs[i][j] indicates the log probability of taking action i at a State env.all_states[j], i in [0, n_actions-1]
    """
    start_time = time.time()
    memo = TensorDict(default_factory=lambda: torch.tensor(['-inf'], requires_grad=False))
    transition_log_probs = get_all_transition_log_probs(env, gfn.pf)
    log_probs = []
    log_probs_termination = []
    # Calculate the log probability and log reward for each terminal state
    # for traj in test_trajectories:
    for terminal_state in tqdm(terminal_states, desc="Evaluating test set...") if show_progress else terminal_states:
        log_prob = compute_log_probability(env, gfn, terminal_state, memo, transition_log_probs)
        log_probs.append(log_prob.detach().numpy())
        log_prob_termination = compute_log_prob_termination(env, terminal_state, memo, transition_log_probs)
        log_probs_termination.append(log_prob_termination.detach().numpy())
    # 9 - Compute Spearman's Rank Correlation
    spearman_corr_termination, _ = spearmanr(log_probs_termination, log_rewards.detach())
    if show_progress:
        print(
        f"Spearman's Rank Correlation (Modified GFNEvalS, including termination actions): {spearman_corr_termination}. Runtime: {time.time() - start_time} seconds.")
    return spearman_corr_termination, memo, transition_log_probs


@timer
def evaluate_GFNEvalS_with_monte_carlo(gfn: GFlowNet,  env: DiscreteStates,
                                       terminal_states, log_rewards, sampler=None,
                                       n_samples=80000,
                                       show_progress: bool = True):
    """Computes the sampling probability of given terminal states using Monte Carlo.

    Args:
        gfn: An initialized Sampler using the forward policy estimator.
        env: The HyperGrid environment instance.
        terminal_states: terminal states in test set.
        log_rewards: true rewards of terminal states in test set.
        n_samples: Generate a large number (n_samples) of samples as monte carlo to count occurrences of appeared terminal states

    Returns:
        spearman_corr_termination: Spearman's Rank Correlation (Modified GFNEvalS, including termination actions)
        occurrences: occurrences dict of each state.tensor appearing in monte carlo
        log_probs_monte_carlo: log probability of given terminal states computed by MC
    """
    start_time = time.time()

    if sampler is None:
        sampler = Sampler(estimator=gfn.pf)
    # Generate a large number of samples as monte carlo to count occurrences of appeared terminal states
    occurrences = count_occurrences_with_monte_carlo(env, sampler, n_samples=n_samples, show_progress=show_progress)
    #
    log_probs_monte_carlo = []

    for terminal_state in tqdm(terminal_states,
                               desc="Evaluating GFNEvalS with monte carlo") if show_progress else terminal_states:
        log_prob = compute_log_prob_with_monte_carlo(occurrences, terminal_state, n_samples)
        log_probs_monte_carlo.append(log_prob.detach().numpy())
    # Compute Spearman's Rank Correlation
    spearman_corr_monte_carlo, _ = spearmanr(log_probs_monte_carlo, log_rewards)
    if show_progress:
        print(
            f"Spearman's Rank Correlation (Monte Carlo): {spearman_corr_monte_carlo}. MC sample number: {n_samples}. Runtime: {time.time() - start_time} seconds")
    return spearman_corr_monte_carlo, occurrences, log_probs_monte_carlo


# Simple learnable function to approximate KL-divergence
class PhiFunction(nn.Module):
    def __init__(self, input_size, layer_size) -> None:
        super(PhiFunction, self).__init__()
        self.linear1 = nn.Linear(input_size, layer_size).to(torch.double)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(layer_size, 1).to(torch.double)

    def forward(self, x):
        # x = x.view(-1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)


def calc_KL_using_model(model, samples_p, samples_q, no_grad=False):
    # Compute f(x) for samples from P and Q
    if no_grad:
        with torch.no_grad():
            f_p = model(samples_p)  # Output shape: [batch_size, 1]
            f_q = model(samples_q)  # Output shape: [batch_size, 1]
    else:
        f_p = model(samples_p)  # Output shape: [batch_size, 1]
        f_q = model(samples_q)  # Output shape: [batch_size, 1]

    # Compute the terms of the formula
    term_p = torch.mean(f_p)  # Expectation over P: E_P[f]
    term_q = torch.log(
        torch.mean(torch.exp(torch.clamp(f_q, max=695, min=-695))))  # Log of expectation over Q: log(E_Q[e^f])

    # KL divergence
    kl_div = term_p - term_q
    return kl_div


def compute_KL(p_star_sample: Tensor, p_hat_sample: Tensor,
               layer_size=128, num_epochs=200, lr=0.001, show_progress=False,
               device='cuda'):
    # Ensure both samples have the same shape
    assert p_star_sample[0].shape == p_hat_sample[0].shape
    input_size = p_star_sample[0].numel()
    # print(input_size)
    # The function to learn
    phi = PhiFunction(input_size=input_size, layer_size=layer_size)
    phi = phi.to(device)
    optimizer = optim.Adam(phi.parameters(), lr=lr)
    # Learn the model
    for epoch in range(num_epochs):
        # Compute KL divergence
        kl_div = calc_KL_using_model(phi, p_star_sample, p_hat_sample)

        # The loss is the negation of the KL-divergence
        loss = -kl_div

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 and show_progress:
            print(f"Epoch {epoch + 1}/{num_epochs}, KL Divergence Estimate: {kl_div:.4f}")

    return kl_div, phi
