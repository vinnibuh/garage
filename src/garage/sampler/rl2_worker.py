"""Visual RL2 Worker class."""
from collections import defaultdict

import numpy as np

from garage import EpisodeBatch, StepType
from garage.sampler.default_worker import DefaultWorker


class VisualRL2Worker(DefaultWorker):
    """Initialize a worker.

    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length (int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        n_episodes_per_trial (int): Number of episodes sampled per
            trial/meta-batch. Policy resets in the beginning of a meta batch,
            and obtain `n_episodes_per_trial` episodes in one meta batch.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number,
            n_episodes_per_trial=2):
        self._n_episodes_per_trial = n_episodes_per_trial
        self._prev_action = None
        self._prev_reward = None
        self._prev_step_type = None
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        
        self._prev_obs, _ = self.env.reset()
        self._prev_action = np.zeros(self.env.action_space.shape)
        self._prev_reward = 0
        self._prev_term_sign = 0

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs,
                                                  self._prev_action,
                                                  self._prev_reward,
                                                  self._prev_term_sign)
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.terminal:
                self._prev_obs = es.observation
                self._prev_action = es.action
                self._prev_reward = es.reward
                self._prev_step_type = 0
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def collect_episode(self):
        """Collect the current episode, clearing the internal buffer.

        Returns:
            EpisodeBatch: A batch of the episodes completed since the last call
                to collect_episode().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []

        actions = []
        rewards = []
        env_infos = defaultdict(list)
        step_types = []

        for es in self._env_steps:
            actions.append(es.action)
            rewards.append(es.reward)
            step_types.append(es.step_type)
            for k, v in es.env_info.items():
                env_infos[k].append(v)
        self._env_steps = []

        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)

        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)

        episode_infos = self._episode_infos
        self._episode_infos = defaultdict(list)
        for k, v in episode_infos.items():
            episode_infos[k] = np.asarray(v)

        lengths = self._lengths
        self._lengths = []
        return EpisodeBatch(env_spec=self.env.spec,
                            episode_infos=episode_infos,
                            observations=np.asarray(observations),
                            last_observations=np.asarray(last_observations),
                            actions=np.asarray(actions),
                            rewards=np.asarray(rewards),
                            step_types=np.asarray(step_types, dtype=StepType),
                            env_infos=dict(env_infos),
                            agent_infos=dict(agent_infos),
                            lengths=np.asarray(lengths, dtype='i'))

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.agent.reset()
        for _ in range(self._n_episodes_per_trial):
            self.start_episode()
            while not self.step_episode():
                pass
        self._agent_infos['batch_idx'] = np.full(len(self._env_steps),
                                                 self._worker_number)
        return self.collect_episode()

    def shutdown(self):
        """Close the worker's environment."""
        self.env.close()
