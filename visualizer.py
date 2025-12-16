"""Compact visualization of game state, learning curve, and network activations."""
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Linear_QNet

class NNVisualizer:
    """Visualize neural network layer activations with network topology"""
    
    def __init__(self, model: Linear_QNet):
        self.model = model
        self.device = model.device
        self.activations = {}
        self.fig = None
        self.ax = None
        
        # Register hooks to capture activations
        self.model.linear1.register_forward_hook(self._hook_fn('hidden'))
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook
    
    def forward_and_visualize(self, state):
        """Forward pass and return activations"""
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(state_tensor)
        
        return output.cpu(), self.activations
    
    def plot_combined_dashboard(self, state, plot_scores, plot_mean_scores, game_num, game=None, title="SnakeAI Dashboard"):
        """Create dashboard with game board + training chart + network topology"""
        output, activations = self.forward_and_visualize(state)
        
        # Create grid layout: left=game, bottom-right=chart, top-right=network
        fig = plt.figure(figsize=(20, 10), facecolor='#0a0e27')
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        # Left panel (spans 2 rows): Game board
        ax_game = fig.add_subplot(gs[:, 0])
        ax_game.set_facecolor('#0a0e27')
        ax_game.set_xlim(-1, 401)
        ax_game.set_ylim(-1, 401)
        ax_game.set_aspect('equal')
        ax_game.invert_yaxis()
        ax_game.axis('off')
        
        # Render game board if game object provided
        if game is not None:
            # Draw grid
            for i in range(0, 400, 20):
                ax_game.plot([i, i], [0, 400], color='#1a3a1a', linewidth=0.5, alpha=0.3)
                ax_game.plot([0, 400], [i, i], color='#1a3a1a', linewidth=0.5, alpha=0.3)
            
            # Draw food
            food = game.food
            ax_game.add_patch(plt.Rectangle((food.x - 10, food.y - 10), 20, 20, 
                                           facecolor='#ff3333', edgecolor='#ff6666', linewidth=2))
            
            # Draw snake
            head = game.snake[0]
            for i, segment in enumerate(game.snake):
                if i == 0:  # Head
                    ax_game.add_patch(plt.Rectangle((segment.x - 10, segment.y - 10), 20, 20,
                                                   facecolor='#00ff88', edgecolor='#00ffcc', linewidth=2.5))
                else:  # Body
                    intensity = 1.0 - (i / len(game.snake))
                    color = f'#{int(intensity * 255):02x}{int(255 * intensity):02x}{int(150 * intensity):02x}'
                    ax_game.add_patch(plt.Rectangle((segment.x - 10, segment.y - 10), 20, 20,
                                                   facecolor=color, edgecolor='#00ff88', linewidth=1, alpha=0.7))
        
        ax_game.set_title('GAME', fontsize=14, color='#00ff88', fontweight='bold', pad=10)
        
        # Bottom-right panel: Training progress chart
        ax_chart = plt.subplot(gs[1, 1])
        ax_chart.set_facecolor('#1a1f3a')
        if plot_scores:
            ax_chart.plot(plot_scores, linewidth=2.5, color='#00ff88', label='Current Score', alpha=0.9, marker='o', markersize=3)
            ax_chart.plot(plot_mean_scores, linewidth=3, color='#0088ff', label='Mean Score (avg)', alpha=0.9)
            ax_chart.fill_between(range(len(plot_scores)), plot_scores, alpha=0.15, color='#00ff88')
            
            # Add max score line
            max_score = max(plot_scores) if plot_scores else 0
            ax_chart.axhline(y=max_score, color='#ffff00', linestyle='--', linewidth=2, alpha=0.7, label=f'Record: {max_score}')
        
        ax_chart.set_xlabel('Game Number', fontsize=11, color='#00ff88', fontweight='bold')
        ax_chart.set_ylabel('Score', fontsize=11, color='#00ff88', fontweight='bold')
        ax_chart.set_title(f'Training Progress (Total Games: {game_num})', fontsize=12, color='#00ff88', fontweight='bold')
        ax_chart.grid(True, alpha=0.25, color='#00ff88', linestyle='--')
        ax_chart.legend(loc='upper left', fontsize=10, facecolor='#1a1f3a', edgecolor='#00ff88', framealpha=0.9)
        ax_chart.tick_params(colors='#00ff88')
        
        # Top-right panel: Network topology
        ax_net = plt.subplot(gs[0, 1])
        ax_net.set_facecolor('#0a0e27')
        ax_net.set_xlim(-1, 11)
        ax_net.set_ylim(-1, 12)
        ax_net.axis('off')
        
        # Network structure
        input_size = 11
        hidden_size = 256
        output_size = 3
        
        # Get layer data
        state_array = np.array(state).astype(float)
        hidden_activations = activations['hidden'][0].numpy()
        output_array = output[0].numpy()
        
        # Normalize for visualization
        state_norm = state_array / (state_array.max() + 1e-6)
        hidden_norm = np.clip(hidden_activations / (hidden_activations.max() + 1e-6), 0, 1)
        output_norm = np.clip(output_array / (output_array.max() + 1e-6), 0, 1)
        
        # Layer positions
        layer_x = [0, 5, 10]
        layer_y_positions = [
            np.linspace(11, 0, input_size),
            np.linspace(11, 0, min(hidden_size, 20)),
            np.linspace(10, 2, output_size)
        ]
        
        # Plot connections
        weights_1 = self.model.linear1.weight.data.cpu().numpy()
        weights_2 = self.model.linear2.weight.data.cpu().numpy()
        
        # Input to Hidden
        for i in range(min(input_size, 11)):
            for j in range(min(hidden_size, 20)):
                weight = weights_1[j, i]
                weight_norm = np.tanh(weight) / 2 + 0.5
                x_vals = [layer_x[0], layer_x[1]]
                y_vals = [layer_y_positions[0][i], layer_y_positions[1][j]]
                alpha = 0.2 + 0.4 * abs(weight_norm)
                color = '#00ff88' if weight > 0 else '#ff0088'
                ax_net.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=0.5)
        
        # Hidden to Output
        for j in range(min(hidden_size, 20)):
            for k in range(output_size):
                weight = weights_2[k, j]
                weight_norm = np.tanh(weight) / 2 + 0.5
                x_vals = [layer_x[1], layer_x[2]]
                y_vals = [layer_y_positions[1][j], layer_y_positions[2][k]]
                alpha = 0.4 + 0.6 * abs(weight_norm)
                color = '#00ff88' if weight > 0 else '#ff0088'
                ax_net.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=1.2)
        
        # Plot input nodes
        ax_net.text(-0.7, 11.5, 'INPUT', fontsize=11, color='#00ff88', fontweight='bold')
        for i, y in enumerate(layer_y_positions[0]):
            size = 80 + 150 * state_norm[i]
            ax_net.scatter(layer_x[0], y, s=size, c=['#00ff88'], alpha=0.8, edgecolors='#00ff88', linewidth=1.5)
        
        # Plot hidden nodes
        ax_net.text(5, 11.8, f'HIDDEN (256 neurons)', fontsize=11, color='#0088ff', fontweight='bold', ha='center')
        for j, y in enumerate(layer_y_positions[1]):
            size = 40 + 80 * hidden_norm[j]
            ax_net.scatter(layer_x[1], y, s=size, c=['#0088ff'], alpha=0.7, edgecolors='#0088ff', linewidth=1.2)
        
        # Plot output nodes
        ax_net.text(10.7, 11.5, 'OUTPUT', fontsize=11, color='#ffff00', fontweight='bold')
        action_names = ['Straight', 'Right', 'Left']
        for k, (y, name, val) in enumerate(zip(layer_y_positions[2], action_names, output_array)):
            size = 150 + 250 * output_norm[k]
            color = '#ffff00' if k == output_array.argmax() else '#ff8800'
            ax_net.scatter(layer_x[2], y, s=size, c=[color], alpha=0.9, edgecolors=color, linewidth=2)
            ax_net.text(11.5, y, f'{name}\n{val:.2f}', fontsize=9, color=color, fontweight='bold', va='center')
        
        fig.suptitle(title, fontsize=16, color='#00ff88', fontweight='bold', y=0.97)
        fig.subplots_adjust(top=0.93, wspace=0.3, hspace=0.3)
        return fig
