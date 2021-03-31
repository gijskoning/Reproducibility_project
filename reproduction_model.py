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

    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(IAMPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            print("discrete")
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            print("Box")
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            print("MultiBinary")
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
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
    def __init__(self, recurrent, recurrent_input_size, hidden_size, recurrent_hidden_size=128):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._recurrent_hidden_size = recurrent_hidden_size

        if recurrent:
            self.lstm = True
            self.rnn = self._create_rnn(recurrent_input_size, recurrent_hidden_size)

    def _create_rnn(self, recurrent_input_size, recurrent_hidden_size):
        # gru = nn.GRU(recurrent_input_size, recurrent_hidden_size)
        rnn = nn.LSTM(recurrent_input_size, recurrent_hidden_size)
        for name, param in rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        return rnn

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            # Changed this! Before it was self._hidden_size
            return self._recurrent_hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_rnn(self, x, hxs, masks, rnn=None):
        if rnn is None:
            # Not used in the IAM model
            rnn = self.rnn

        if x.size(0) == hxs.size(0) or (self.lstm and x.size(0) == hxs.size(0)/2):
            print((hxs * masks).unsqueeze(1).shape)
            print((hxs * masks).unsqueeze(1).unsqueeze(1).shape)
            print(x.unsqueeze(0).shape)
            if self.lstm:
                x, hxs = rnn(x.unsqueeze(0), (hxs * masks).unsqueeze(1).unsqueeze(1))
            else:
                x, hxs = rnn(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            if self.lstm:
                hxs = hxs.squeeze(1).squeeze(1)
            else:
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

                rnn_scores, hxs = rnn(
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
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, second_hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, second_hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        def create_base():
            return nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, second_hidden_size)), nn.Tanh())

        self.actor = create_base()

        self.critic = create_base()

        self.critic_linear = init_(nn.Linear(second_hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class IAMBase(MLPBase):
    def __init__(self, num_inputs, recurrent, hidden_size, second_hidden_size):
        super(IAMBase, self).__init__(num_inputs, recurrent, hidden_size, second_hidden_size)
        assert recurrent

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        def create_base():
            return nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, second_hidden_size)), nn.Tanh())

        recurrent_input_size = num_inputs
        self.rnn_hidden_size = 128
        self.actor = create_base()
        self.actor_rnn = self._create_rnn(recurrent_input_size, self.rnn_hidden_size)
        # self.actor_linear_combine_rnn = init_(nn.Linear(second_hidden_size, 1))

        self.critic = create_base()
        self.critic_rnn = self._create_rnn(recurrent_input_size, self.rnn_hidden_size)
        # self.critic_linear_combine_rnn = init_(nn.Linear(second_hidden_size, 1))

        self.critic_linear = init_(nn.Linear(second_hidden_size + self.rnn_hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        # Split the rnn_hxs in two. This is just a hack to get two rnn's at the same time with the ppo algo!

        left_rnn_hxs = rnn_hxs[:, :self._recurrent_hidden_size]
        right_rnn_hxs = rnn_hxs[:, self._recurrent_hidden_size:self._recurrent_hidden_size * 2]

        if self.lstm:
            left_state_cell = rnn_hxs[:, self._recurrent_hidden_size * 2:self._recurrent_hidden_size * 3]
            right_state_cell = rnn_hxs[:, self._recurrent_hidden_size * 3:]

            left_rnn_hxs = torch.cat([left_rnn_hxs, left_state_cell])
            right_rnn_hxs = torch.cat([right_rnn_hxs, right_state_cell])
        critic_x, left_rnn_hxs = \
            self._forward_rnn(x, left_rnn_hxs, masks, self.actor_rnn)

        actor_x, right_rnn_hxs = \
            self._forward_rnn(x, right_rnn_hxs, masks, self.critic_rnn)

        hidden_critic = torch.cat([hidden_critic, critic_x], 1)
        hidden_actor = torch.cat([hidden_actor, actor_x], 1)

        if self.lstm:
            right_rnn_hxs, right_cell_state = right_rnn_hxs
            # unpack hxs
            left_rnn_hxs, left_cell_state = left_rnn_hxs
            rnn_hxs[:, self._recurrent_hidden_size * 2:self._recurrent_hidden_size * 3] = left_cell_state
            rnn_hxs[:, self._recurrent_hidden_size * 3:self._recurrent_hidden_size * 4] = right_cell_state

        rnn_hxs[:, :self._recurrent_hidden_size] = left_rnn_hxs[0]
        rnn_hxs[:, self._recurrent_hidden_size:self._recurrent_hidden_size * 2] = right_rnn_hxs[0]

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

    @property
    def recurrent_hidden_state_size(self):
        # Changed this! Before it was self._hidden_size
        # Do this hack multiplying by such that one half is used by the critic network and the other by the actor.
        size = self._recurrent_hidden_size * 2
        if (self.lstm):
            size *= 2
        return size

    @property
    def output_size(self):
        return self._hidden_size + self.rnn_hidden_size
