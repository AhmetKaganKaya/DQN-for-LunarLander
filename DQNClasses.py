import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Network(nn.Module):
    def __init__(self, input_dim, fc1, fc2, n_actions,lr):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, fc1)
        self.layer2 = nn.Linear(fc1,fc2)
        self.layer3 = nn.Linear(fc2, n_actions)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        actions = self.layer3(x)

        return actions

class Agent():
    def __init__(self, gamma, eps, lr, n_actions, max_mem, batch_size, obs_dim):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.mem_index = 0
        self.mem_total = 0
        self.max_mem = max_mem
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.network = Network(input_dim=self.obs_dim, fc1= 256, fc2 = 256, n_actions= n_actions, lr= self.lr).double()

        self.optimizer = optim.Adam(params=self.network.parameters(), lr = lr)
        self.loss = nn.MSELoss()


        self.state_memory = np.zeros((self.max_mem, obs_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem, obs_dim), dtype=np.float32)

        self.action_memory = np.zeros(self.max_mem, dtype= np.int32)
        self.reward_memory = np.zeros(self.max_mem, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem, dtype=np.int32)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_index % self.max_mem
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = int(done)

        self.mem_total += 1
        self.mem_index += 1
        self.mem_total = np.minimum(self.mem_total, self.max_mem)

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = T.tensor([observation]).double().to(self.network.device)
            actions = self.network.forward(state.unsqueeze(0))
            action = int(T.argmax(actions))
        else:
            action = np.random.choice(self.n_actions)
        return  action

    def learn(self):
        if self.mem_total < self.batch_size:
            return

        self.optimizer.zero_grad()
        batch = np.random.choice(self.mem_total, self.batch_size, replace= False)

        state_batch = T.tensor(self.state_memory[batch]).double().to(self.network.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).double().to(self.network.device)
        reward_batch = T.tensor(self.reward_memory[batch]).double().to(self.network.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).double().to(self.network.device)
        action_batch = T.tensor(self.action_memory[batch]).double().to(self.network.device)

        q_eval = self.network.forward(state_batch).gather(1, action_batch.unsqueeze(1).long())
        q_next = self.network.forward(new_state_batch).detach().max(1)[0].unsqueeze(1)

        q_target = reward_batch.unsqueeze(-1) + self.gamma * (1 - terminal_batch.unsqueeze(-1)) * q_next

        loss = self.loss(q_target,q_eval).to(self.network.device)
        loss.backward()
        self.optimizer.step()
















