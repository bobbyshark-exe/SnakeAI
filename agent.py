"""DQN agent orchestrating multi-env training and live metrics capture."""
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from visualizer import NNVisualizer

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
        # Load pre-trained model and memory
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load pre-trained model weights and replay memory"""
        # Load model
        if self.model.load():
            print("✓ Loaded pre-trained model")
        
        # Load memory
        memory_file = './model/memory.pkl'
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'rb') as f:
                    saved_memory = pickle.load(f)
                    self.memory = deque(saved_memory, maxlen=MAX_MEMORY)
                    print(f"✓ Loaded replay memory with {len(self.memory)} experiences")
            except Exception as e:
                print(f"Could not load memory: {e}")
    
    def save_checkpoint(self):
        """Save model weights and replay memory"""
        # Memory is saved whenever model is saved
        if not os.path.exists('./model'):
            os.makedirs('./model')
        memory_file = './model/memory.pkl'
        with open(memory_file, 'wb') as f:
            pickle.dump(list(self.memory), f)

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
        # More aggressive exploration early on
        self.epsilon = max(0, 200 - self.n_games)  # Explore for first 200 games
        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1 # Random exploration
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 # Use learned model
            
        return final_move

def train(num_envs=3, visual=True):
    import signal
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    games = [SnakeGameAI(visual=(visual and idx == 0)) for idx in range(num_envs)]
    visualizer = NNVisualizer(agent.model)
    
    def save_on_exit(sig, frame):
        """Save memory and model on Ctrl+C"""
        print("\n\n💾 Saving checkpoint on exit...")
        agent.save_checkpoint()
        print(f"✓ Saved at Game {agent.n_games}")
        exit(0)
    
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, save_on_exit)

    while True:
        states_old = [agent.get_state(g) for g in games]
        moves = [agent.get_action(s) for s in states_old]

        step_results = [g.play_step(a) for g, a in zip(games, moves)]
        rewards, dones, scores = zip(*step_results)
        states_new = [agent.get_state(g) for g in games]

        # Train short memory in batch
        agent.train_short_memory(states_old, moves, rewards, states_new, dones)
        # Remember batch
        for s0, a, r, s1, d in zip(states_old, moves, rewards, states_new, dones):
            agent.remember(s0, a, r, s1, d)

        # Handle episodes that ended; reset and bookkeeping per env
        for i, done in enumerate(dones):
            if done:
                games[i].reset()
                agent.n_games += 1
                agent.train_long_memory()
                score = scores[i]
                if score > record:
                    record = score
                    agent.model.save()
                    agent.save_checkpoint()  # Save memory with best model
                # Save memory periodically (every 50 games)
                if agent.n_games % 50 == 0:
                    agent.save_checkpoint()
                print('Game', agent.n_games, 'Score', score, 'Record:', record)
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                # Update live dashboard after every game (game parameter for live game rendering)
                fig = visualizer.plot_combined_dashboard(
                    states_old[0], 
                    plot_scores, 
                    plot_mean_scores, 
                    agent.n_games,
                    game=games[i],  # Pass the game object for live rendering
                    title=f"SnakeAI Live Dashboard - Game {agent.n_games} | Record: {record} | Score: {score} | Avg: {mean_score:.1f}"
                )
                plt.savefig('dashboard.png', dpi=80, bbox_inches='tight', facecolor='#0a0e27')
                plt.close(fig)
                
                # Also save training chart for reference (legacy)
                if agent.n_games % 10 == 0:
                    plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    # Run 3 parallel simulations; only the first is visual for a clean window
    train(num_envs=3, visual=True)