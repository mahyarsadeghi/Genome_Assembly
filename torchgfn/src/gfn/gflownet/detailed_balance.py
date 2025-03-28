import math
from typing import Tuple

import torch

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.modules import ConditionalScalarEstimator, GFNModule, ScalarEstimator
from gfn.utils.common import has_log_probs
from gfn.utils.handlers import (
    has_conditioning_exception_handler,
    no_conditioning_exception_handler,
)


def check_compatibility(states, actions, transitions):
    if states.batch_shape != tuple(actions.batch_shape):
        if type(transitions) is not Transitions:
            raise TypeError(
                "`transitions` is type={}, not Transitions".format(type(transitions))
            )
        else:
            raise ValueError(" wrong happening with log_pf evaluations")


class DBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""The Detailed Balance GFlowNet.

    Corresponds to $\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times
    \mathcal{O}_3$, where $\mathcal{O}_1$ is the set of functions from the internal
    states (no $s_f$) to $\mathbb{R}^+$ (which we parametrize with logs, to avoid the
    non-negativity constraint), and $\mathcal{O}_2$ is the set of forward probability
    functions consistent with the DAG. $\mathcal{O}_3$ is the set of backward
    probability functions consistent with the DAG, or a singleton thereof, if
    `self.logit_PB` is a fixed `DiscretePBEstimator`.

    Attributes:
        logF: a ScalarEstimator instance.
        forward_looking: whether to implement the forward looking GFN loss.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logF: ScalarEstimator | ConditionalScalarEstimator,
        forward_looking: bool = False,
        log_reward_clip_min: float = -float("inf"),
    ):
        super().__init__(pf, pb)
        assert any(
            isinstance(logF, cls)
            for cls in [ScalarEstimator, ConditionalScalarEstimator]
        ), "logF must be a ScalarEstimator or derived"
        self.logF = logF
        self.forward_looking = forward_looking
        self.log_reward_clip_min = log_reward_clip_min

    def logF_named_parameters(self):
        try:
            return {k: v for k, v in self.named_parameters() if "logF" in k}
        except KeyError as e:
            print(
                "logF not found in self.named_parameters. Are the weights tied with PF? {}".format(
                    e
                )
            )

    def logF_parameters(self):
        try:
            return [v for k, v in self.named_parameters() if "logF" in k]
        except KeyError as e:
            print(
                "logF not found in self.named_parameters. Are the weights tied with PF? {}".format(
                    e
                )
            )

    def get_scores(
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Returns: A tuple of three tensors of shapes (n_transitions,), representing the
            log probabilities of the actions, the log probabilities of the backward actions, and th scores.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)
        check_compatibility(states, actions, transitions)

        if has_log_probs(transitions) and not recalculate_all_logprobs:
            valid_log_pf_actions = transitions.log_probs
        else:
            # Evaluate the log PF of the actions, with optional conditioning.
            # TODO: Inefficient duplication in case of tempered policy
            # The Transitions container should then have some
            # estimator_outputs attribute as well, to avoid duplication here ?
            # See (#156).
            if transitions.conditioning is not None:
                with has_conditioning_exception_handler("pf", self.pf):
                    module_output = self.pf(states, transitions.conditioning)
            else:
                with no_conditioning_exception_handler("pf", self.pf):
                    module_output = self.pf(states)

            valid_log_pf_actions = self.pf.to_probability_distribution(
                states, module_output
            ).log_prob(actions.tensor)

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s = self.logF(states, transitions.conditioning).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s = self.logF(states).squeeze(-1)

        if self.forward_looking:
            log_rewards = env.log_reward(states)  # TODO: RM unsqueeze(-1) ?
            if math.isfinite(self.log_reward_clip_min):
                log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
            valid_log_F_s = valid_log_F_s + log_rewards

        preds = valid_log_pf_actions + valid_log_F_s
        targets = torch.zeros_like(preds)

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_actions = actions[~actions.is_exit]

        # Evaluate the log PB of the actions, with optional conditioning.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(
                    valid_next_states, transitions.conditioning[~transitions.is_done]
                )
        else:
            with no_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(valid_next_states)

        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        # LogF is potentially a conditional computation.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(
                    valid_next_states, transitions.conditioning[~transitions.is_done]
                ).squeeze(-1)
        else:
            with no_conditioning_exception_handler("logF", self.logF):
                valid_log_F_s_next = self.logF(valid_next_states).squeeze(-1)

        targets[~valid_transitions_is_done] = valid_log_pb_actions
        log_pb_actions = targets.clone()
        targets[~valid_transitions_is_done] += valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ]

        scores = preds - targets

        assert valid_log_pf_actions.shape == (transitions.n_transitions,)
        assert log_pb_actions.shape == (transitions.n_transitions,)
        assert scores.shape == (transitions.n_transitions,)
        return valid_log_pf_actions, log_pb_actions, scores

    def loss(self, env: Env, transitions: Transitions) -> torch.Tensor:
        """Detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)."""
        _, _, scores = self.get_scores(env, transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()


class ModifiedDBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""The Modified Detailed Balance GFlowNet. Only applicable to environments where
    all states are terminating.

    See Bayesian Structure Learning with Generative Flow Networks
    https://arxiv.org/abs/2202.13903 for more details.
    """

    def get_scores(
        self, transitions: Transitions, recalculate_all_logprobs: bool = False
    ) -> torch.Tensor:
        """DAG-GFN-style detailed balance, when all states are connected to the sink.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Raises:
            ValueError: when backward transitions are supplied (not supported).
            ValueError: when the computed scores contain `inf`.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")

        mask = ~transitions.next_states.is_sink_state
        states = transitions.states[mask]
        valid_next_states = transitions.next_states[mask]
        actions = transitions.actions[mask]
        all_log_rewards = transitions.all_log_rewards[mask]

        check_compatibility(states, actions, transitions)

        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(states, transitions.conditioning[mask])
        else:
            with no_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(states)

        pf_dist = self.pf.to_probability_distribution(states, module_output)

        if has_log_probs(transitions) and not recalculate_all_logprobs:
            valid_log_pf_actions = transitions[mask].log_probs
        else:
            # Evaluate the log PF of the actions sampled off policy.
            valid_log_pf_actions = pf_dist.log_prob(actions.tensor)
        valid_log_pf_s_exit = pf_dist.log_prob(
            torch.full_like(actions.tensor, actions.__class__.exit_action[0])
        )

        # The following two lines are slightly inefficient, given that most
        # next_states are also states, for which we already did a forward pass.
        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(
                    valid_next_states, transitions.conditioning[mask]
                )
        else:
            with no_conditioning_exception_handler("pf", self.pf):
                module_output = self.pf(valid_next_states)

        valid_log_pf_s_prime_exit = self.pf.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(torch.full_like(actions.tensor, actions.__class__.exit_action[0]))

        non_exit_actions = actions[~actions.is_exit]

        if transitions.conditioning is not None:
            with has_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(
                    valid_next_states, transitions.conditioning[mask]
                )
        else:
            with no_conditioning_exception_handler("pb", self.pb):
                module_output = self.pb(valid_next_states)

        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        preds = all_log_rewards[:, 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
        targets = all_log_rewards[:, 1] + valid_log_pb_actions + valid_log_pf_s_exit

        scores = preds - targets
        if torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")

        return scores

    def loss(self, env: Env, transitions: Transitions) -> torch.Tensor:
        """Calculates the modified detailed balance loss."""
        scores = self.get_scores(transitions)
        return torch.mean(scores**2)

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
