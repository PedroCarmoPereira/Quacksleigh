import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG) and Proximal Policy Optimization (PPO) 
# Paper: https://arxiv.org/abs/1509.02971


class ActorDense(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorDense, self).__init__()

        state_dim = functools.reduce(operator.mul, state_dim, 1)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

        self.tanh = nn.Tanh()

        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * self.tanh(self.l3(x))
        return x

    def get_distribution(self, x):
        mean = self.forward(x)
        action_std = self.action_log_std.exp().expand_as(mean)
        return Normal(mean, action_std)

class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action):
        super(ActorCNN, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        # this is the vanilla implementation
        # but we're using a slightly different one
        # x = self.max_action * self.tanh(self.lin2(x))

        # because we don't want our duckie to go backwards
        # x = self.lin2(x)
        # x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        # x[:, 1] = self.tanh(x[:, 1])

        v = self.max_action * self.sigm(x[:, 0:1])  # because we don't want the duckie to go backwards
        w = self.tanh(x[:, 1:2])

        #return x
        return torch.cat([v, w], dim=1)

    def get_distribution(self, x):
        mean = self.forward(x)
        action_std = self.action_log_std.exp().expand_as(mean)
        return Normal(mean, action_std)


class CriticDense(nn.Module):
    #def __init__(self, state_dim, action_dim):
    def __init__(self, state_dim):
        super(CriticDense, self).__init__()

        state_dim = functools.reduce(operator.mul, state_dim, 1)

        self.l1 = nn.Linear(state_dim, 400)
        #self.l2 = nn.Linear(400 + action_dim, 300)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    #def forward(self, x, u):
    #    x = F.relu(self.l1(x))
    #    x = F.relu(self.l2(torch.cat([x, u], 1)))
    #    x = self.l3(x)
    #    return x
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticCNN(nn.Module):
    #def __init__(self, action_dim):
    def __init__(self):
        super(CriticCNN, self).__init__()

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(flat_size, 256)
        #self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        #x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lr(self.lin2(x))
        x = self.lin3(x)

        return x

class PPO(object):
    def __init__(self, state_dim, action_dim, max_action, net_type):
        super(PPO, self).__init__()
        print("Starting PPO init")
        assert net_type in ["cnn", "dense"]

        self.state_dim = state_dim
        self.action_dim = action_dim

        if net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.critic = CriticDense(state_dim).to(device)
        else:
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action).to(device)
            self.critic = CriticCNN().to(device)

        print("Initialized Actor and Critic")
        
        # PPO doesn't use target networks, just optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        print("Initialized Optimizers")

    def predict(self, state):
        """Used for deterministic evaluation / real-world Duckiebot deployment"""
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
            
        with torch.no_grad():
            mean_action = self.actor(state)
            
        return mean_action.cpu().data.numpy().flatten()
        
    def select_action(self, state):
        """Used for stochastic trajectory collection during simulation training"""
        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
            
        with torch.no_grad():
            dist = self.actor.get_distribution(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(-1)
            
        return action.cpu().data.numpy().flatten(), action_logprob.item()

    def train(self, rollout_buffer, epochs=10, batch_size=64, clip_ratio=0.2, discount=0.99):
        """
        PPO is on-policy. rollout_buffer should be a dict/object containing
        the full trajectories collected by the CURRENT policy.
        """
        states = torch.FloatTensor(rollout_buffer["states"]).to(device)
        actions = torch.FloatTensor(rollout_buffer["actions"]).to(device)
        old_log_probs = torch.FloatTensor(rollout_buffer["log_probs"]).to(device)
        returns = torch.FloatTensor(rollout_buffer["returns"]).to(device)
        advantages = torch.FloatTensor(rollout_buffer["advantages"]).to(device)
        
        # Normalize advantages to improve training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)

        for _ in range(epochs):
            # Shuffle data for mini-batch updates
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # 1. Evaluate current policy against old actions
                dist = self.actor.get_distribution(batch_states)
                state_values = self.critic(batch_states).squeeze(-1)
                
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1)

                # 2. Calculate PPO Ratio
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # 3. Surrogate Loss for Actor
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                # 4. MSE Loss for Critic
                critic_loss = F.mse_loss(state_values, batch_returns)

                # 5. Optimize Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 6. Optimize Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), "{}/{}_actor.pth".format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic.state_dict(), "{}/{}_critic.pth".format(directory, filename))
        print("Saved Critic")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("{}/{}_actor.pth".format(directory, filename), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load("{}/{}_critic.pth".format(directory, filename), map_location=device)
        )


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, net_type):
        super(DDPG, self).__init__()
        print("Starting DDPG init")
        assert net_type in ["cnn", "dense"]

        self.state_dim = state_dim

        if net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorDense(state_dim, action_dim, max_action).to(device)
        else:
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action).to(device)
            self.actor_target = ActorCNN(action_dim, max_action).to(device)

        print("Initialized Actor")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        print("Initialized Target+Opt [Actor]")
        if net_type == "dense":
            self.critic = CriticDense(state_dim, action_dim).to(device)
            self.critic_target = CriticDense(state_dim, action_dim).to(device)
        else:
            self.critic = CriticCNN(action_dim).to(device)
            self.critic_target = CriticCNN(action_dim).to(device)
        print("Initialized Critic")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        print("Initialized Target+Opt [Critic]")

    def predict(self, state):

        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), "{}/{}_actor.pth".format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic.state_dict(), "{}/{}_critic.pth".format(directory, filename))
        print("Saved Critic")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("{}/{}_actor.pth".format(directory, filename), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load("{}/{}_critic.pth".format(directory, filename), map_location=device)
        )
