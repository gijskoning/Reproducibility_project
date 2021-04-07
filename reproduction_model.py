import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class IAMPolicy(nn.Module):
    """
    This class contains the reproduction Policy model.
    """

    def __init__(self, obs_shape, action_space, IAM=False, RNN=False, base_kwargs=None):
        super(IAMPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.IAM = IAM
        self.recurrent = RNN

        if len(obs_shape) == 3:
            if self.IAM:
                print("Using IAMBaseCNN")
                base = IAMBaseCNN
            else:
                print("Using CNNBase")
                base = CNNBase
        elif len(obs_shape) == 1:
            if self.IAM:
                print("Using IAMBase")
                base = IAMBase
            else:
                if self.recurrent:
                    print("Using RNNBase")
                    base = RNNBase
                else:
                    print("Using MLPBase")
                    base = MLPBase
        else:
            raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            print("discrete")
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size(), num_outputs)
        elif action_space.__class__.__name__ == "Box":
            print("Box")
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size(), num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            print("MultiBinary")
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size(), num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, hidden_size, recurrent_hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._recurrent_hidden_size = recurrent_hidden_size

        if recurrent:
            self.gru = self._create_gru(hidden_size, recurrent_hidden_size)

    def _create_gru(self, recurrent_input_size, recurrent_hidden_size):
        gru = nn.GRU(recurrent_input_size, recurrent_hidden_size)
        for name, param in gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        return gru

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            # Changed this! Before it was self._hidden_size
            return self._recurrent_hidden_size
        return 1

    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks, gru=None):
        # if gru is None:
        # Not used in the IAM model
        # gru = self.gru

        if x.size(0) == hxs.size(0):
            x, hxs = gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_sizes=(64, 64), recurrent_hidden_size=128):
        super(MLPBase, self).__init__(recurrent, hidden_sizes[-1], recurrent_hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor_fnn = self.create_fnn(num_inputs, hidden_sizes)

        self.critic_fnn = self.create_fnn(num_inputs, hidden_sizes)

        self.critic_linear = init_(nn.Linear(hidden_sizes[-1], 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        # This is not used
        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic_fnn(x)
        hidden_actor = self.actor_fnn(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

    def create_fnn(self, num_inputs, hidden_sizes):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        seq_list = [init_(nn.Linear(num_inputs, hidden_sizes[0])), nn.Tanh()]
        if len(hidden_sizes) > 1:
            seq_list.extend([init_(nn.Linear(*hidden_sizes)), nn.Tanh()])

        module = nn.Sequential(*seq_list)
        return module


class IAMBase(MLPBase):
    def __init__(self, num_inputs, recurrent, hidden_sizes, rnn_input_size=None, rnn_hidden_size=25):
        super(IAMBase, self).__init__(num_inputs, recurrent, hidden_sizes, rnn_hidden_size)
        assert recurrent
        # todo could remove this if statement since currently input size is always none
        if rnn_input_size is None:
            rnn_input_size = num_inputs

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        # self.static_A_matrix = init_(nn.Linear(num_inputs, rnn_input_size))

        self.actor_rnn = self._create_gru(rnn_input_size, self._recurrent_hidden_size)

        self.critic_rnn = self._create_gru(rnn_input_size, self._recurrent_hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.critic_linear = init_(nn.Linear(hidden_sizes[-1] + self._recurrent_hidden_size, 1))

        self.train()

    def forward(self, input, rnn_hxs, masks):
        fnn_input = input
        rnn_input = input

        hidden_critic = self.critic_fnn(fnn_input)
        hidden_actor = self.actor_fnn(fnn_input)
        # Split the rnn_hxs in two. This is just a hack to get two GRU's at the same time with the ppo algo!
        left_rnn_hxs, right_rnn_hxs = rnn_hxs.split(self._recurrent_hidden_size, 1)

        hidden_critic_rnn, left_rnn_hxs = \
            self._forward_gru(rnn_input, left_rnn_hxs, masks, self.actor_rnn)

        hidden_actor_rnn, right_rnn_hxs = \
            self._forward_gru(rnn_input, right_rnn_hxs, masks, self.critic_rnn)

        # Combine critic FNN with RNN
        hidden_critic = torch.cat([hidden_critic, hidden_critic_rnn], 1)
        hidden_actor = torch.cat([hidden_actor, hidden_actor_rnn], 1)

        rnn_hxs = torch.cat([left_rnn_hxs, right_rnn_hxs], 1)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

    @property
    def recurrent_hidden_state_size(self):
        # Changed this! Before it was self._hidden_size
        # Do this hack multiplying by such that one half is used by the critic network and the other by the actor.
        return self._recurrent_hidden_size * 2

    def output_size(self):
        return self._hidden_size + self._recurrent_hidden_size


class IAMBaseCNN(IAMBase):

    def __init__(self, num_inputs, recurrent, hidden_sizes, rnn_hidden_size):
        final_hidden_size = 64
        final_hidden_size_flattened = final_hidden_size * 7 * 7
        # to much hacking here
        super(IAMBaseCNN, self).__init__(final_hidden_size_flattened, recurrent, hidden_sizes,
                                         rnn_hidden_size=rnn_hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.cnn_preprocessor = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, (8, 8), stride=(4, 4))), nn.ReLU(),
            init_(nn.Conv2d(32, 64, (4, 4), stride=(2, 2))), nn.ReLU(),
            init_(nn.Conv2d(64, final_hidden_size, (3, 3), stride=(1, 1))), nn.ReLU(), Flatten())

    def forward(self, input, rnn_hxs, masks):
        """
        This method preprocesses the input with a CNN. Then filters the input with the static A matrix (Linear layer).
        The processed input with the static_d_set output is passed forward to the regular IAMModel where the FNN and RNN are.
        """
        processed_input = self.cnn_preprocessor(input)

        return super(IAMBaseCNN, self).forward(processed_input, rnn_hxs, masks)


class RNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_sizes=(64, 64), recurrent_hidden_size=128):
        super(RNNBase, self).__init__(recurrent, hidden_sizes[-1], recurrent_hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor_rnn = self._create_gru(num_inputs, self._recurrent_hidden_size)

        self.critic_rnn = self._create_gru(num_inputs, self._recurrent_hidden_size)

        self.critic_linear = init_(nn.Linear(hidden_sizes[-1], 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        left_rnn_hxs, right_rnn_hxs = rnn_hxs.split(self._recurrent_hidden_size, 1)

        hidden_critic_rnn, left_rnn_hxs = \
            self._forward_gru(x, left_rnn_hxs, masks, self.actor_rnn)

        hidden_actor_rnn, right_rnn_hxs = \
            self._forward_gru(x, right_rnn_hxs, masks, self.critic_rnn)

        rnn_hxs = torch.cat([left_rnn_hxs, right_rnn_hxs], 1)

        return self.critic_linear(hidden_critic_rnn), hidden_actor_rnn, rnn_hxs

    @property
    def recurrent_hidden_state_size(self):
        # Changed this! Before it was self._hidden_size
        # Do this hack multiplying by such that one half is used by the critic network and the other by the actor.
        return self._recurrent_hidden_size * 2

    def output_size(self):
        return self._recurrent_hidden_size