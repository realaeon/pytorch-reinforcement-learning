
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,N_STATES,N_ACTIONS ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 5)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(5, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        actions_value = self.out(x)
        return actions_value#F.softmax(actions_value)

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax()

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, self.action_size)

  def forward(self, x, h):
    x = self.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h


