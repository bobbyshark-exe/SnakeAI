import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

#  1. THE NEURAL NETWORK (The Brain) 
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Use CPU for now (RTX 5060 Blackwell not yet fully supported)
        # Once RTX 5060 drivers/PyTorch mature, switch to: torch.device('cuda')
        self.device = torch.device('cpu')
        # Layer 1: Input (11) -> Hidden (256)
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        # Layer 2: Hidden (256) -> Output (3)
        self.linear2 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        # Activation Function: ReLU (Rectified Linear Unit)
        # It turns negative numbers to 0 (firing/not firing logic)
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x) # No activation on last layer (raw Q-values)
        return x

    def save(self, file_name='model.pth'):
        # Helper to save the trained brain so we don't lose progress
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
    
    def load(self, file_name='model.pth'):
        # Load pre-trained model weights
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            return True
        return False

#  2. THE TRAINER (The Teacher) 
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # Learning Rate (How fast it changes beliefs)
        self.gamma = gamma # Discount Rate (Care about future vs immediate reward)
        self.model = model
        self.device = model.device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

    def train_step(self, state, action, reward, next_state, done):
        # Convert lists to numpy arrays first for faster tensor creation
        import numpy as np
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)

        # Handle formatting for single step vs batch training
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        #  THE BELLMAN EQUATION IMPLEMENTATION 
        
        # 1. Predicted Q values with current state
        # "What do I THINK the value of this move is?"
        pred = self.model(state)

        target = pred.clone()
        # Compute argmax indices of actions once
        act_idx = torch.argmax(action, dim=1)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][act_idx[idx].item()] = Q_new
    
        # 3. Backpropagation
        # "Adjust the brain weights to make Prediction closer to Target"
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # Calculate error
        loss.backward() # Calculate gradients
        self.optimizer.step() # Update weights