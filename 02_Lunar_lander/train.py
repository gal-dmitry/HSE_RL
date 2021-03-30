from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 4)
)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change

        self.model = copy.deepcopy(model)
        self.target_model = copy.deepcopy(model)

        self.model.cuda()
        self.target_model.cuda()

        self.replay_buffer = deque(maxlen=INITIAL_STEPS)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

        #         device = torch.device("cuda")
        #         self.model.to(device)
        #         self.target_model.to(device)

        # self.model.cuda()
        # self.target_model.cuda()

    #         self.optimizer.cuda()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.

        state, action, next_state, reward, done = transition

        action_ = [0 for i in range(8)]
        action_[0] = action

        reward_ = [0 for i in range(8)]
        reward_[0] = reward

        done_ = [0 for i in range(8)]
        if done:
            done_[0] = 1

        self.replay_buffer.append([state, action_, next_state, reward_, done_])

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        batch = np.array(batch, dtype=np.float32).reshape(BATCH_SIZE, 5, -1)
        batch = torch.from_numpy(batch).cuda()
        return batch

    def train_step(self, batch):
        # Use batch to update DQN's network.

        ### transition
        state = batch[:, 0]
        action = batch[:, 1, 0].type(torch.LongTensor).cuda()
        next_state = batch[:, 2]
        reward = batch[:, 3, 0]
        done = batch[:, 4, 0].type(torch.LongTensor).cuda()

        #         print(state.get_device())
        #         print(action.get_device())
        #         print(next_state.get_device())
        #         print(reward.get_device())
        #         print(done.get_device())

        ### target network
        q_target = torch.zeros(reward.size()[0]).float()

        with torch.no_grad():
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target[done == 1] = 0

        q_target = (reward + q_target * GAMMA).unsqueeze(1)

        ### model network
        q_model = self.model(state).gather(1, action.unsqueeze(1))

        ### Loss
        loss = F.mse_loss(q_model, q_target)

        ### step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.tensor(state).cuda()
        state = state.float().unsqueeze(0)
        action = self.model(state)[0].max(0)[1].view(1, 1).item()
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1

    env.seed(0)
    np.random.seed(0)
    random.seed(0)

    state = env.reset()

    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()