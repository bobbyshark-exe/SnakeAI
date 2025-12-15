import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# --- 1. THE NEURAL NETWORK (The Brain) ---
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Layer 1: Input (11) -> Hidden (256)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Layer 2: Hidden (256) -> Output (3)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Activation Function: ReLU (Rectified Linear Unit)
        # It turns negative numbers to 0 (firing/not firing logic)
        x = F.relu(self.linear1(x))
        x = self.linear2(x) # No activation on last layer (raw Q-values)
        return x

    def save(self, file_name='model.pth'):
        # Helper to save the trained brain so we don't lose progress
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# --- 2. THE TRAINER (The Teacher) ---
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # Learning Rate (How fast it changes beliefs)
        self.gamma = gamma # Discount Rate (Care about future vs immediate reward)
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

    def train_step(self, state, action, reward, next_state, done):
        # Convert numpy arrays to PyTorch tensors (math format)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle formatting for single step vs batch training
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # --- THE BELLMAN EQUATION IMPLEMENTATION ---
        
        # 1. Predicted Q values with current state
        # "What do I THINK the value of this move is?"
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 2. Q_new = Reward + (Gamma * Max(Next_State_Q))
                # "The actual value is the immediate reward + the best future reward"
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the target Q-value for the action we actually took
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 3. Backpropagation
        # "Adjust the brain weights to make Prediction closer to Target"
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # Calculate error
        loss.backward() # Calculate gradients
        self.optimizer.step() # Update weights