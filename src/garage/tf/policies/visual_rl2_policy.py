"""Visual RL2 Policy.

A combined policy for Visual RL2,
using CNN for images and RNN for 
joint embeddings.
"""
# pylint: disable=wrong-import-order
import akro
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models.cnn_gru_merge_model import CNNGRUMergeModel
from garage.tf.policies.policy import Policy

# pylint: disable=too-many-ancestors
class VisualRL2Policy(CNNGRUMergeModel, Policy):
    """Visual RL2 Policy.

    A combined policy for Visual RL2, using CNN for images and RNN for 
    joint embeddings.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Model name, also the variable scope.
        hidden_dim (int): Hidden dimension for GRU cell for mean.
        hidden_nonlinearity (Callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (Callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (Callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        recurrent_nonlinearity (Callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        recurrent_w_init (Callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (Callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (Callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (Callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        hidden_state_init (Callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        learn_std (bool): Is std trainable.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        layer_normalization (bool): Bool for using layer normalization or not.
        state_include_action (bool): Whether the state includes action.
            If True, input dimension will be
            (observation dimension + action dimension).

    """

    def __init__(self,
                 env_spec,
                 cnn_filters,
                 cnn_strides,
                 mlp_hidden_sizes=(256,),
                 action_emb_size=256,
                 gru_hidden_dim=256,
                 name='CustomRNNRL2Policy',
                 cnn_padding='SAME',
                 cnn_max_pooling=False,
                 cnn_pool_strides=(2, 2),
                 cnn_pool_shapes=(2, 2),
                 cnn_hidden_nonlinearity=tf.nn.relu,
                 cnn_hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 cnn_hidden_b_init=tf.zeros_initializer(),
                 mlp_hidden_nonlinearity=tf.nn.relu,
                 mlp_hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 mlp_hidden_b_init=tf.zeros_initializer(),
                 mlp_output_nonlinearity=None,
                 mlp_output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 mlp_output_b_init=tf.zeros_initializer(),
                 mlp_layer_normalization=False,
                 gru_hidden_nonlinearity=tf.nn.tanh,
                 gru_hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 gru_hidden_b_init=tf.zeros_initializer(),
                 gru_recurrent_nonlinearity=tf.nn.sigmoid,
                 gru_recurrent_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 gru_output_nonlinearity=None,
                 gru_output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 gru_output_b_init=tf.zeros_initializer(),
                 gru_hidden_state_init=tf.zeros_initializer(),
                 gru_hidden_state_init_trainable=False,
                 gru_layer_normalization=False,
                 gru_learn_std=True,
                 gru_std_share_network=False,
                 gru_init_std=1.0):
        if not isinstance(env_spec.action_space, akro.Box):
            raise ValueError('GaussianGRUPolicy only works with '
                             'akro.Box action space, but not {}'.format(
                                 env_spec.action_space))

        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.shape
        self._action_dim = env_spec.action_space.flat_dim

        self._cnn_filters = cnn_filters
        self._cnn_strides = cnn_strides
        self._cnn_padding = cnn_padding
        self._mlp_hidden_sizes = mlp_hidden_sizes
        self._action_emb_size = action_emb_size
        self._gru_hidden_dim = gru_hidden_dim
        self._cnn_max_pooling = cnn_max_pooling
        self._cnn_pool_strides = cnn_pool_strides
        self._cnn_pool_shapes = cnn_pool_shapes
        self._cnn_hidden_nonlinearity = cnn_hidden_nonlinearity
        self._cnn_hidden_w_init = cnn_hidden_w_init
        self._cnn_hidden_b_init = cnn_hidden_b_init
        self._mlp_hidden_nonlinearity = mlp_hidden_nonlinearity
        self._mlp_hidden_w_init = mlp_hidden_w_init
        self._mlp_hidden_b_init = mlp_hidden_b_init
        self._mlp_output_nonlinearity = mlp_output_nonlinearity
        self._mlp_output_w_init = mlp_output_w_init
        self._mlp_output_b_init = mlp_output_b_init
        self._mlp_layer_normalization = mlp_layer_normalization
        self._gru_hidden_nonlinearity = gru_hidden_nonlinearity
        self._gru_hidden_w_init = gru_hidden_w_init
        self._gru_hidden_b_init = gru_hidden_b_init
        self._gru_recurrent_nonlinearity = gru_recurrent_nonlinearity
        self._gru_recurrent_w_init = gru_recurrent_w_init
        self._gru_output_nonlinearity = gru_output_nonlinearity
        self._gru_output_w_init = gru_output_w_init
        self._gru_output_b_init = gru_output_b_init
        self._gru_hidden_state_init = gru_hidden_state_init
        self._gru_hidden_state_init_trainable = gru_hidden_state_init_trainable
        self._gru_layer_normalization = gru_layer_normalization
        self._gru_learn_std = gru_learn_std
        self._gru_std_share_network = gru_std_share_network
        self._gru_init_std = gru_init_std

        self._input_dim = self._obs_dim

        self._f_step_mean_std = None

        super().__init__(
            cnn_input_dim=self._obs_dim,
            output_dim=self._action_dim,
            cnn_filters=self._cnn_filters,
            cnn_strides = self._cnn_strides,
            mlp_hidden_sizes=self._mlp_hidden_sizes,
            action_emb_size=self._action_emb_size,
            gru_hidden_dim=self._gru_hidden_dim,
            name=name,
            cnn_padding=self._cnn_padding,
            cnn_max_pooling=self._cnn_max_pooling,
            cnn_pool_strides=self._cnn_pool_strides,
            cnn_pool_shapes=self._cnn_pool_shapes,
            cnn_hidden_nonlinearity=self._cnn_hidden_nonlinearity,
            cnn_hidden_w_init=self._cnn_hidden_w_init,
            cnn_hidden_b_init=self._cnn_hidden_b_init,
            mlp_hidden_nonlinearity=self._mlp_hidden_nonlinearity,
            mlp_hidden_w_init=self._mlp_hidden_w_init,
            mlp_hidden_b_init=self._mlp_hidden_b_init,
            mlp_output_nonlinearity=self._mlp_output_nonlinearity,
            mlp_output_w_init=self._mlp_output_w_init,
            mlp_output_b_init=self._mlp_output_b_init,
            mlp_layer_normalization=self._mlp_layer_normalization,
            gru_hidden_nonlinearity=self._gru_hidden_nonlinearity,
            gru_hidden_w_init=self._gru_hidden_w_init,
            gru_hidden_b_init=self._gru_hidden_b_init,
            gru_recurrent_nonlinearity=self._gru_recurrent_nonlinearity,
            gru_recurrent_w_init=self._gru_recurrent_w_init,
            gru_output_nonlinearity=self._gru_output_nonlinearity,
            gru_output_w_init=self._gru_output_w_init,
            gru_output_b_init=self._gru_output_b_init,
            gru_hidden_state_init=self._gru_hidden_state_init,
            gru_hidden_state_init_trainable=self._gru_hidden_state_init_trainable,
            gru_layer_normalization=self._gru_layer_normalization,
            gru_learn_std=self._gru_learn_std,
            gru_std_share_network=self._gru_std_share_network,
            gru_init_std=self._gru_init_std)

        self._prev_actions = None
        self._prev_hiddens = None
        self._init_hidden = None

        self._initialize_policy()

    def _initialize_policy(self):
        """Initialize policy."""
        full_obs_input = tf.compat.v1.placeholder(shape=(None, None,
                                                    *self._obs_dim),
                                             name='obs_input',
                                             dtype=tf.float32)
        full_action_input = tf.compat.v1.placeholder(shape=(None, None,
                                                       self._action_dim),
                                                name='action_input',
                                                dtype=tf.float32)
        full_reward_input = tf.compat.v1.placeholder(shape=(None, None),
                                                name='reward_input',
                                                dtype=tf.float32)
        full_term_sign_input = tf.compat.v1.placeholder(shape=(None, None),
                                                   name='term_sign_input',
                                                   dtype=tf.float32)
        step_obs_input = tf.compat.v1.placeholder(shape=(None,
                                                    *self._obs_dim),
                                             name='obs_input',
                                             dtype=tf.float32)
        step_action_input = tf.compat.v1.placeholder(shape=(None,
                                                       self._action_dim),
                                                name='action_input',
                                                dtype=tf.float32)
        step_reward_input = tf.compat.v1.placeholder(shape=(None, ),
                                                name='reward_input',
                                                dtype=tf.float32)
        step_term_sign_input = tf.compat.v1.placeholder(shape=(None, ),
                                                   name='term_sign_input',
                                                   dtype=tf.float32)
        step_hidden_var = tf.compat.v1.placeholder(shape=(None,
                                                          self._gru_hidden_dim),
                                                   name='step_hidden_input',
                                                   dtype=tf.float32)
        (_, step_mean, step_log_std, step_hidden,
         self._init_hidden) = super().build(
             full_obs_input, full_action_input, full_reward_input, full_term_sign_input,
             step_obs_input, step_action_input, step_reward_input, step_term_sign_input,                                            
             step_hidden_var).outputs

        self._f_step_mean_std = (
            tf.compat.v1.get_default_session().make_callable(
                [step_mean, step_log_std, step_hidden],
                feed_list=[step_obs_input, step_action_input,
                           step_reward_input, step_term_sign_input,
                           step_hidden_var]))

    def build(self, full_obs_input, full_action_input, full_reward_input, full_term_sign_input, name=None):
        """Build policy.

        Args:
            state_input (tf.Tensor) : State input.
            name (str): Name of the policy, which is also the name scope.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Policy distribution.
            tf.Tensor: Step means, with shape :math:`(N, S^*)`.
            tf.Tensor: Step log std, with shape :math:`(N, S^*)`.
            tf.Tensor: Step hidden state, with shape :math:`(N, S^*)`.
            tf.Tensor: Initial hidden state, with shape :math:`(S^*)`.

        """
        _, _, _, _, step_obs_var, step_action_var, step_reward_var, step_term_sign_var, step_hidden_var = self.inputs
        return super().build(
            full_obs_input, full_action_input, full_reward_input, full_term_sign_input,
            step_obs_var, step_action_var, step_reward_var, step_term_sign_var,
            step_hidden_var, name=name)

    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._input_dim

    def reset(self, do_resets=None):
        """Reset the policy.

        Note:
            If `do_resets` is None, it will be by default `np.array([True])`
            which implies the policy will not be "vectorized", i.e. number of
            parallel environments for training data sampling = 1.

        Args:
            do_resets (numpy.ndarray): Bool that indicates terminal state(s).

        """
        if do_resets is None:
            do_resets = np.array([True])
        if self._prev_actions is None or len(do_resets) != len(
                self._prev_actions):
            self._prev_actions = np.zeros(
                (len(do_resets), self.action_space.flat_dim))
            self._prev_hiddens = np.zeros((len(do_resets), self._gru_hidden_dim))

        self._prev_actions[do_resets] = 0.
        self._prev_hiddens[do_resets] = self._init_hidden.eval()

    def get_action(self, observation, action, reward, term_sign):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Mean of the distribution.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution.
            - prev_action (numpy.ndarray): Previous action, only present if
                self._state_include_action is True.

        """
        actions, agent_infos = self.get_actions([observation], 
                                                [action],
                                                [reward],
                                                [term_sign])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations, actions, rewards, term_signs):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Means of the distribution.
            - log_std (numpy.ndarray): Log standard deviations of the
                distribution.
            - prev_action (numpy.ndarray): Previous action, only present if
                self._state_include_action is True.

        """
        means, log_stds, hidden_vec = self._f_step_mean_std(
            observations, actions, rewards, term_signs, self._prev_hiddens)
        rnd = np.random.normal(size=means.shape)
        samples = rnd * np.exp(log_stds) + means
        samples = self.action_space.unflatten_n(samples)
        self._prev_actions = samples
        self._prev_hiddens = hidden_vec
        agent_infos = dict(mean=means, log_std=log_stds)
        return samples, agent_infos

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def clone(self, name):
        """Return a clone of the policy.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created policy. It has to be
                different from source policy if cloned under the same
                computational graph.

        Returns:
            garage.tf.policies.GaussianGRUPolicy: Newly cloned policy.

        """
        new_policy = self.__class__(
            name=name,
            env_spec=self._env_spec,
            cnn_filters=self._cnn_filters,
            cnn_strides=self._cnn_strides,
            mlp_hidden_sizes=self._mlp_hidden_sizes,
            action_emb_size=self._action_emb_size,
            gru_hidden_dim=self._gru_hidden_dim,
            cnn_padding=self._cnn_padding,
            cnn_max_pooling=self._cnn_max_pooling,
            cnn_pool_strides=self._cnn_pool_strides,
            cnn_pool_shapes=self._cnn_pool_shapes,
            cnn_hidden_nonlinearity=self._cnn_hidden_nonlinearity,
            cnn_hidden_w_init=self._cnn_hidden_w_init,
            cnn_hidden_b_init=self._cnn_hidden_b_init,
            mlp_hidden_nonlinearity=self._mlp_hidden_nonlinearity,
            mlp_hidden_w_init=self._mlp_hidden_w_init,
            mlp_hidden_b_init=self._mlp_hidden_b_init,
            mlp_output_nonlinearity=self._mlp_output_nonlinearity,
            mlp_output_w_init=self._mlp_output_w_init,
            mlp_output_b_init=self._mlp_output_b_init,
            mlp_layer_normalization=self._mlp_layer_normalization,
            gru_hidden_nonlinearity=self._gru_hidden_nonlinearity,
            gru_hidden_w_init=self._gru_hidden_w_init,
            gru_hidden_b_init=self._gru_hidden_b_init,
            gru_recurrent_nonlinearity=self._gru_recurrent_nonlinearity,
            gru_recurrent_w_init=self._gru_recurrent_w_init,
            gru_output_nonlinearity=self._gru_output_nonlinearity,
            gru_output_w_init=self._gru_output_w_init,
            gru_output_b_init=self._gru_output_b_init,
            gru_hidden_state_init=self._gru_hidden_state_init,
            gru_hidden_state_init_trainable=self._gru_hidden_state_init_trainable,
            gru_layer_normalization=self._gru_layer_normalization,
            gru_learn_std=self._gru_learn_std,
            gru_std_share_network=self._gru_std_share_network,
            gru_init_std=self._gru_init_std)
        new_policy.parameters = self.parameters
        return new_policy

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_step_mean_std']
        del new_dict['_init_hidden']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize_policy()
