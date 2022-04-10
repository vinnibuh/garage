"""CNN and MLP Merge Model."""
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models.cnn_model import CNNModel
from garage.tf.models.cnn_model_max_pooling import CNNModelWithMaxPooling
from garage.tf.models.gaussian_gru_model import GaussianGRUModel
from garage.tf.models.mlp_model import MLPModel
from garage.tf.models.model import Model


class CNNGRUMergeModel(Model):
    """Convolutional neural network followed by a Multilayer Perceptron.

    Combination of a CNN Model (optionally with max pooling) and an
    MLP Merge model. The CNN accepts the state as an input, while
    the MLP accepts the CNN's output and the action as inputs.

    Args:
        input_dim (Tuple[int, int, int]): Dimensions of unflattened input,
            which means [in_height, in_width, in_channels]. If the last 3
            dimensions of input_var is not this shape, it will be reshaped.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_sizes (tuple[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        output_dim (int): Dimension of the network output.
        action_merge_layer (int): The index of layers at which to concatenate
            action inputs with the network. The indexing works like standard
            python list indexing. Index of 0 refers to the input layer
            (observation input) while an index of -1 points to the last
            hidden layer. Default parameter points to second layer from the
            end.
        name (str): Model name, also the variable scope.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pooling (bool): Boolean for using max pooling layer or not.
        pool_shapes (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_strides (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        cnn_hidden_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the CNN. It should return a
            tf.Tensor. Set it to None to maintain a linear activation.
        cnn_hidden_w_init (callable): Initializer function for the weight of
            intermediate dense layer(s) in the CNN. Function should return a
            tf.Tensor.
        cnn_hidden_b_init (callable): Initializer function for the bias of
            intermediate dense layer(s) in the CNN. Function should return a
            tf.Tensor.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s) in the MLP. It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s) in the MLP. The function should
            return a tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s) in the MLP. The function should
            return a tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer in the MLP. It should return a tf.Tensor. Set it to None
            to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the MLP. The function should return
            a tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s) in the MLP. The function should return
            a tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 cnn_input_dim,
                 output_dim,
                 cnn_filters,
                 cnn_strides,
                 mlp_hidden_sizes=(256, ),
                 action_emb_size=256,
                 gru_hidden_dim=32,
                 name="CNNGRUMergeModel",
                 cnn_padding='SAME',
                 cnn_max_pooling=False,
                 cnn_pool_strides=(2, 2),
                 cnn_pool_shapes=(2, 2),
                 cnn_hidden_nonlinearity=tf.nn.relu,
                 cnn_hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 cnn_hidden_b_init=tf.zeros_initializer(),
                 mlp_hidden_nonlinearity=tf.nn.relu,
                 mlp_hidden_w_init=tf.initializers.orthogonal(
                     seed=deterministic.get_tf_seed_stream()),
                 mlp_hidden_b_init=tf.zeros_initializer(),
                 mlp_output_nonlinearity=None,
                 mlp_output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 mlp_output_b_init=tf.zeros_initializer(),
                 mlp_layer_normalization=True,
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
        super().__init__(name)

        if not cnn_max_pooling:
            self.cnn_model = CNNModel(
                name="observation_cnn",
                input_dim=cnn_input_dim,
                filters=cnn_filters,
                hidden_w_init=cnn_hidden_w_init,
                hidden_b_init=cnn_hidden_b_init,
                strides=cnn_strides,
                padding=cnn_padding,
                hidden_nonlinearity=cnn_hidden_nonlinearity)
        else:
            self.cnn_model = CNNModelWithMaxPooling(
                name="observation_cnn",
                input_dim=cnn_input_dim,
                filters=cnn_filters,
                hidden_w_init=cnn_hidden_w_init,
                hidden_b_init=cnn_hidden_b_init,
                strides=cnn_strides,
                padding=cnn_padding,
                pool_strides=cnn_pool_strides,
                pool_shapes=cnn_pool_shapes,
                hidden_nonlinearity=cnn_hidden_nonlinearity)

        self.mlp_action_model = MLPModel(
            name="action_mlp",
            output_dim=action_emb_size,
            hidden_sizes=mlp_hidden_sizes,
            hidden_nonlinearity=mlp_hidden_nonlinearity,
            hidden_w_init=mlp_hidden_w_init,
            hidden_b_init=mlp_hidden_b_init,
            output_nonlinearity=mlp_output_nonlinearity,
            output_w_init=mlp_output_w_init,
            output_b_init=mlp_output_b_init,
            layer_normalization=mlp_layer_normalization)

        self.mlp_joint_model = MLPModel(
            name="joint_mlp",
            output_dim=output_dim,
            hidden_sizes=mlp_hidden_sizes,
            hidden_nonlinearity=mlp_hidden_nonlinearity,
            hidden_w_init=mlp_hidden_w_init,
            hidden_b_init=mlp_hidden_b_init,
            output_nonlinearity=mlp_output_nonlinearity,
            output_w_init=mlp_output_w_init,
            output_b_init=mlp_output_b_init,
            layer_normalization=mlp_layer_normalization)

        self.gru_model = GaussianGRUModel(
            name="state_gru",
            output_dim=output_dim,
            hidden_dim=gru_hidden_dim,
            hidden_nonlinearity=gru_hidden_nonlinearity,
            hidden_w_init=gru_hidden_w_init,
            hidden_b_init=gru_hidden_b_init,
            recurrent_nonlinearity=gru_recurrent_nonlinearity,
            recurrent_w_init=gru_recurrent_w_init,
            output_nonlinearity=gru_output_nonlinearity,
            output_w_init=gru_output_w_init,
            output_b_init=gru_output_b_init,
            hidden_state_init=gru_hidden_state_init,
            hidden_state_init_trainable=gru_hidden_state_init_trainable,
            layer_normalization=gru_layer_normalization,
            learn_std=gru_learn_std,
            std_share_network=gru_std_share_network,
            init_std=gru_init_std
        )

    def network_input_spec(self):
        """Network input spec.

        Return:
            list[str]: List of key(str) for the network inputs.

        """
        return ['full_obs_input', 'full_action_input', 'full_reward_input', 'full_term_sign_input', 
                'step_obs_input', 'step_action_input', 'step_reward_input', 'step_term_sign_input', 
                'step_hidden_input']

    def network_output_spec(self):
        """Network output spec.

        Returns:
            list[str]: Name of the model outputs, in order.

        """
        return [
            'dist', 'step_mean', 'step_log_std', 'step_hidden', 'init_hidden'
        ]

    # pylint: disable=arguments-differ
    def _build(self, obs, action, reward, term_sign,
                     obs_step, action_step, reward_step, term_sign_step,
                     gru_hidden_input, name=None):
        """Build the model and return the outputs.

        This builds the model such that the output of the CNN is fed
        to the MLP. The CNN receives the state as the input. The MLP
        receives two inputs, the output of the CNN and the action
        tensor.

        Args:
            obs (tf.Tensor): Obs placeholder tensor of shape
                :math:`(N, O*)`.
            action (tf.Tensor): Action placeholder tensor of shape
                :math:`(N, A*)`.
            reward (tf.Tensor): Reward placeholder tensor of shape
                :math:`(N, T, 1)`.
            term_sigm (tf.Tensor): Termination sign placeholder tensor of shape
                :math:`(N, T, 1)`.
            gru_hidden_input (tf.Tensor): Hidden state for step, with shape
                :math:`(N, S^*)`.
            name (str): Name of the model.

        Returns:
            tf.Tensor: Output of the model of shape (N, output_dim).

        """
        # full input
        time_dim = tf.shape(obs)[1]
        dim = obs.get_shape()[2:].as_list()
        augm_obs = tf.reshape(obs, [-1, *dim])
        cnn_out = self.cnn_model.build(augm_obs,
                                       name=f'{name}_full_cnn').outputs
        dim = cnn_out.get_shape()[-1]
        cnn_out = tf.reshape(cnn_out, [-1, time_dim, dim])

        mlp_action_out = self.mlp_action_model.build(action, name=f'{name}_full_action').outputs

        augm_reward = tf.expand_dims(reward, -1)
        augm_term_sign = tf.expand_dims(term_sign, -1)

        joint_in = tf.concat([cnn_out, mlp_action_out, augm_reward, augm_term_sign], -1)
        mlp_joint_out = self.mlp_joint_model.build(joint_in, name=f'{name}_full_joint').outputs
        # step input
        step_cnn_out = self.cnn_model.build(obs_step, name=f'{name}_step_cnn').outputs
        step_mlp_action_out = self.mlp_action_model.build(action_step, name=f'{name}_step_action').outputs
        augm_reward_step = tf.expand_dims(reward_step, -1)
        augm_term_sign_step = tf.expand_dims(term_sign_step, -1)
        step_joint_in = tf.concat([step_cnn_out, step_mlp_action_out, augm_reward_step, augm_term_sign_step], -1)
        step_mlp_joint_out = self.mlp_joint_model.build(step_joint_in, name=f'{name}_step_joint').outputs
        gru_out = self.gru_model.build(mlp_joint_out, step_mlp_joint_out, gru_hidden_input, 
                                       name=f'{name}_gru').outputs
        return gru_out
