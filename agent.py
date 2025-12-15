import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning Rate

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness control
        self.gamma = 0.9 # Discount rate (must be < 1)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if memory exceeds limit
        self.model = Linear_QNet(11, 256, 3) # 11 Inputs, 256 Hidden, 3 Outputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        
        # Create points around the head to check for danger
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Current direction boolean check
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # THE 11 INPUTS (STATE VECTOR)
        state = [
            # 1. Danger Straight ahead
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # 2. Danger Right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # 3. Danger Left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # 4. Move Direction (One-Hot Encoded)
            dir_l, 
            dir_r, 
            dir_u, 
            dir_d,
            
            # 5. Food Location (relative to head)
            game.food.x < game.head.x,  # Food Left
            game.food.x > game.head.x,  # Food Right
            game.food.y < game.head.y,  # Food Up
            game.food.y > game.head.y   # Food Down
        ]
        
        # Convert True/False to 1/0
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train on a random batch of 1000 previous moves
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on only the move that just happened
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-Greedy Strategy:
        # In the beginning, make random moves to explore.
        # As games increase, make fewer random moves and use the Brain.
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1 # Random Move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 # Predicted Move
            
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # 1. Get current state
        state_old = agent.get_state(game)

        # 2. Get move
        final_move = agent.get_action(state_old)

        # 3. Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train Short Memory (Train on step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Remember (Store in replay memory)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 6. Game Over: Train Long Memory (Replay Experience) & Plot
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()