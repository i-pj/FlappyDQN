import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import sys
sys.path.append("game/")
import game.wrapped as game
import random
import numpy as np
from collections import deque
import wandb

# Parameters
ACTIONS = 2 #jump, do nothing
GAMMA = 0.99 
OBSERVE = 1000 
EXPLORE = 2000000 
FINAL_EPSILON = 0.0001 
INITIAL_EPSILON = 0.1 
REPLAY_MEMORY = 50000 
BATCH = 32 
FRAME_PER_ACTION = 1


# Create model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
    
# Initialize model
model = DQN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

# Initialize replay memory
replay_memory = deque()

# Initialize game
game_state = game.GameState()


# Initialize epsilon
epsilon = INITIAL_EPSILON

# Initialize state
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
x_t, r_0, terminal = game_state.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

# Initialize wandb
wandb.init(project="FlappyBird", entity="parth-jain", config={
    "ACTIONS": ACTIONS,
    "GAMMA": GAMMA,
    "OBSERVE": OBSERVE,
    "EXPLORE": EXPLORE,
    "FINAL_EPSILON": FINAL_EPSILON,
    "INITIAL_EPSILON": INITIAL_EPSILON,
    "REPLAY_MEMORY": REPLAY_MEMORY,
    "BATCH": BATCH,
    "FRAME_PER_ACTION": FRAME_PER_ACTION
})

# Training
for i in range(100000):
    s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0)
    Q = model(s_t)
    a_t = torch.zeros(ACTIONS)
    if random.random() <= epsilon:
        action_index = random.randrange(ACTIONS)
    else:
        action_index = torch.argmax(Q).item()
    a_t[action_index] = 1

    x_t1_colored, r_t, terminal = game_state.frame_step(a_t.cpu().numpy())
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 84, 84))
    s_t1 = np.append(x_t1, s_t[0, 1:, :, :].cpu().numpy(), axis=0)

    replay_memory.append((s_t, action_index, r_t, s_t1, terminal))
    if len(replay_memory) > REPLAY_MEMORY:
        replay_memory.popleft()

    if i > OBSERVE:
        minibatch = random.sample(replay_memory, BATCH)
        s_j_batch = torch.stack([d[0][0] for d in minibatch]).float()
        a_batch = torch.tensor([d[1] for d in minibatch], dtype=torch.int64)
        r_batch = torch.tensor([d[2] for d in minibatch], dtype=torch.float32)
        s_j1_batch = torch.stack([torch.from_numpy(d[3]) for d in minibatch]).float()

        Q = model(s_j_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
        Q_next = model(s_j1_batch).detach()
        max_Q_next = torch.max(Q_next, 1)[0]
        target = r_batch + GAMMA * max_Q_next

        optimizer.zero_grad()
        loss = criterion(Q, target)
        loss.backward()
        optimizer.step()

    s_t = s_t1
    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    wandb.log({"epsilon": epsilon, "reward": r_t})

    if i % 1000 == 0:
        torch.save(model.state_dict(), "model.pth")

# Save model
torch.save(model.state_dict(), "model.pth")

# Close wandb
wandb.finish()