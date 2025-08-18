# GPU-Optimized Enhanced Stratego with Periodic Results and Game Timelapse
# Fixed and completed version

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import math
import os
import json
from datetime import datetime
import imageio
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
BOARD_SIZE = 10
NUM_PIECES = 40
HIDDEN_PIECE = -1
EMPTY_SQUARE = 0
LAKE_SQUARE = -2

# Piece Ranks
FLAG = 0
SPY = 1
SCOUT = 2
MINER = 3
SERGEANT = 4
LIEUTENANT = 5
CAPTAIN = 6
MAJOR = 7
COLONEL = 8
GENERAL = 9
MARSHAL = 10
BOMB = 11

PIECE_NAMES = {
    FLAG: 'F', SPY: '1', SCOUT: '2', MINER: '3', SERGEANT: '4',
    LIEUTENANT: '5', CAPTAIN: '6', MAJOR: '7', COLONEL: '8',
    GENERAL: '9', MARSHAL: 'X', BOMB: 'B', EMPTY_SQUARE: '.',
    LAKE_SQUARE: '~', HIDDEN_PIECE: '?'
}

# Colors for visualization
PIECE_COLORS = {
    FLAG: '#FFD700', SPY: '#FF69B4', SCOUT: '#87CEEB', MINER: '#8B4513',
    SERGEANT: '#32CD32', LIEUTENANT: '#00CED1', CAPTAIN: '#FF6347',
    MAJOR: '#9370DB', COLONEL: '#FF4500', GENERAL: '#DC143C',
    MARSHAL: '#8B0000', BOMB: '#2F4F4F', EMPTY_SQUARE: '#F5F5DC',
    LAKE_SQUARE: '#4169E1', HIDDEN_PIECE: '#696969'
}

# --- GPU-Optimized Data Structures ---

@dataclass
class GameState:
    """Lightweight game state for GPU processing."""
    board: torch.Tensor
    current_player: int
    turn_count: int
    game_over: bool
    winner: Optional[int]
    move_history: List[Tuple]
    uncertainty_mask: torch.Tensor

class MCTSNode:
    """Simplified MCTS node optimized for GPU batch processing."""
    def __init__(self, parent=None, action=None, prior_prob=0.0):
        self.parent = parent
        self.action = action
        self.children: Dict[Tuple, MCTSNode] = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False

    def ucb_score(self, c_puct=1.4):
        """Calculates the Upper Confidence Bound for Trees (UCT) score."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration

    def select_child(self):
        """Selects the child with the highest UCB score."""
        return max(self.children.values(), key=lambda child: child.ucb_score())

    def expand(self, action_probs: List[Tuple[Tuple, float]]):
        """Expands the node by creating children for valid actions."""
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, action=action, prior_prob=prob)
        self.is_expanded = True

    def backup(self, value: float):
        """Backpropagates the simulation result up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            # The value is from the perspective of the player who made the move *into* the state.
            # So, we negate it for the parent.
            node.total_value += value
            value = -value
            node = node.parent

# --- Game Recorder for Timelapse ---

class GameRecorder:
    """Records game states for creating timelapse videos."""
    def __init__(self, episode_num, output_dir="game_timelapses"):
        self.episode_num = episode_num
        self.output_dir = os.path.join(output_dir, f"episode_{episode_num}")
        self.frames = []
        self.actions = []
        self.players = []
        os.makedirs(self.output_dir, exist_ok=True)

    def record_state(self, board: torch.Tensor, current_player: int, action: Optional[Tuple] = None):
        """Records a single frame of the game."""
        # Detach from GPU and copy to CPU for storage
        self.frames.append(board.cpu().numpy().copy())
        self.players.append(current_player)
        self.actions.append(action)

    def create_timelapse(self, fps=2, winner=None):
        """Creates a timelapse GIF from the recorded frames."""
        if not self.frames:
            print("No states recorded for timelapse.")
            return

        print(f"Creating timelapse for episode {self.episode_num} with {len(self.frames)} frames...")
        image_files = []
        for i, (board, player, action) in enumerate(zip(self.frames, self.players, self.actions)):
            frame_img = self._create_board_image(board, player, action, i, winner)
            frame_path = os.path.join(self.output_dir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)
            image_files.append(frame_path)

        # Create GIF from saved frames
        gif_path = os.path.join(os.path.dirname(self.output_dir), f"game_timelapse_episode_{self.episode_num}.gif")
        with imageio.get_writer(gif_path, mode='I', fps=fps, loop=0) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        # Clean up individual frame files
        for filename in image_files:
            os.remove(filename)
        os.rmdir(self.output_dir)
        
        print(f"✅ Timelapse saved: {gif_path}")

    def _create_board_image(self, board, current_player, action, turn_num, winner):
        """Creates a visual representation of the board state as a PIL Image."""
        cell_size = 60
        board_dim = BOARD_SIZE * cell_size
        margin = 50
        info_height = 100
        img_width = board_dim + 2 * margin
        img_height = board_dim + 2 * margin + info_height

        img = Image.new('RGB', (img_width, img_height), '#E0E0E0')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arialbd.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            font_small = font
            
        # Draw board cells and pieces
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1, y1 = margin + c * cell_size, margin + r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                piece_val = int(board[r, c])
                piece_rank = abs(piece_val)
                
                color = PIECE_COLORS.get(piece_rank, '#CCCCCC')
                if piece_val < 0 and piece_rank not in [LAKE_SQUARE, EMPTY_SQUARE]:
                     # Player 2 (Red team) gets a red background tint
                     rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                     blended_color = tuple(int(c1*0.4 + c2*0.6) for c1, c2 in zip(rgb_color, (200, 50, 50)))
                     color = f"#{blended_color[0]:02x}{blended_color[1]:02x}{blended_color[2]:02x}"

                draw.rectangle([x1, y1, x2, y2], fill=color, outline='black')

                if piece_rank in PIECE_NAMES:
                    symbol = PIECE_NAMES.get(piece_rank, '?')
                    text_color = 'white' if piece_val < 0 else 'black'
                    bbox = draw.textbbox((0, 0), symbol, font=font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x1 + (cell_size - text_w) / 2, y1 + (cell_size - text_h) / 2),
                              symbol, fill=text_color, font=font)

        # Highlight last move
        if action:
            (r_from, c_from), (r_to, c_to) = action
            # Highlight destination
            x1, y1 = margin + c_to * cell_size, margin + r_to * cell_size
            draw.rectangle([x1, y1, x1 + cell_size, y1 + cell_size], outline='lime', width=4)
            # Draw arrow
            center_from = (margin + c_from * cell_size + cell_size//2, margin + r_from * cell_size + cell_size//2)
            center_to = (margin + c_to * cell_size + cell_size//2, margin + r_to * cell_size + cell_size//2)
            draw.line([center_from, center_to], fill='purple', width=4)

        # Add game info text
        info_y = margin + board_dim + 20
        player_color = "Blue" if current_player == 1 else "Red"
        info_text = f"Episode: {self.episode_num} | Turn: {turn_num} | Player to move: {player_color}"
        draw.text((margin, info_y), info_text, fill='black', font=font_small)

        if winner is not None:
             winner_color = "Blue (P1)" if winner == 1 else "Red (P2)"
             winner_text = f"GAME OVER! Winner: {winner_color}"
             if turn_num == len(self.frames) - 1:
                draw.text((margin, info_y + 30), winner_text, fill='green', font=font)

        return img

# --- Periodic Results Plotter ---

class PeriodicResultsPlotter:
    """Creates periodic result plots every N episodes."""
    def __init__(self, plot_frequency=100, output_dir="periodic_results"):
        self.plot_frequency = plot_frequency
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def should_plot(self, episode_num):
        """Check if we should create a plot at this episode."""
        return (episode_num + 1) % self.plot_frequency == 0

    def create_periodic_plot(self, episode_num, history):
        """Create and save a comprehensive results plot."""
        print(f"📊 Creating periodic results plot for episode {episode_num + 1}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Episode {episode_num + 1}', fontsize=16, fontweight='bold')

        # Win Rates
        ax = axes[0, 0]
        win_history = history['wins']
        if win_history:
            win_p1 = np.array([1 if w == 1 else 0 for w in win_history])
            win_p2 = np.array([1 if w == -1 else 0 for w in win_history])
            draws = np.array([1 if w == 0 else 0 for w in win_history])
            window = max(1, len(win_history) // 10)
            moving_avg_p1 = np.convolve(win_p1, np.ones(window)/window, mode='valid')
            moving_avg_p2 = np.convolve(win_p2, np.ones(window)/window, mode='valid')
            ax.plot(moving_avg_p1, label=f'Player 1 Wins (MA)', color='blue')
            ax.plot(moving_avg_p2, label=f'Player 2 Wins (MA)', color='red')
        ax.set_title('Win Rate (Moving Average)')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Win Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss
        ax = axes[0, 1]
        loss_history = history['loss']
        if loss_history:
            ax.plot(loss_history, label='Total Loss', color='purple')
            ax.set_yscale('log')
        ax.set_title('Training Loss (Log Scale)')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Game Lengths
        ax = axes[0, 2]
        game_lengths = history['game_lengths']
        if game_lengths:
            window = max(1, len(game_lengths) // 10)
            moving_avg = np.convolve(game_lengths, np.ones(window)/window, mode='valid')
            ax.plot(moving_avg, color='green', label=f'{window}-ep MA')
        ax.set_title('Game Length (Moving Average)')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Number of Turns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Recent Win/Loss Pie Chart
        ax = axes[1, 0]
        recent_games = min(100, len(win_history))
        if recent_games > 0:
            recent_results = win_history[-recent_games:]
            p1_wins = sum(1 for w in recent_results if w == 1)
            p2_wins = sum(1 for w in recent_results if w == -1)
            draws = sum(1 for w in recent_results if w == 0)
            if sum([p1_wins, p2_wins, draws]) > 0:
                ax.pie([p1_wins, p2_wins, draws], labels=['P1 Wins', 'P2 Wins', 'Draws'],
                       colors=['blue', 'red', 'gray'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Last {recent_games} Games')

        # Q-Values
        ax = axes[1, 1]
        q_values = history['q_values']
        if q_values:
            ax.plot(q_values, label='Avg. Root Q-Value', color='orange')
        ax.set_title('MCTS Root Q-Value')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Avg. Q-Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # GPU Memory
        ax = axes[1, 2]
        gpu_mem = history['gpu_mem']
        if gpu_mem:
             ax.plot(gpu_mem, label='GPU Memory (MB)', color='teal')
        ax.set_title('GPU Memory Usage')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Memory (MB)')
        ax.legend()
        ax.grid(True, alpha=0.3)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filepath = os.path.join(self.output_dir, f"results_episode_{episode_num + 1}.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"✅ Periodic plot saved: {filepath}")

# --- Action Space Helpers ---

def generate_all_possible_moves(board_size):
    """Generates all possible moves and creates index mappings."""
    moves = []
    # Normal moves (1 step horizontal/vertical)
    for r in range(board_size):
        for c in range(board_size):
            if r > 0: moves.append(((r, c), (r - 1, c)))
            if r < board_size - 1: moves.append(((r, c), (r + 1, c)))
            if c > 0: moves.append(((r, c), (r, c - 1)))
            if c < board_size - 1: moves.append(((r, c), (r, c + 1)))
    
    # Scout moves (straight lines)
    for r in range(board_size):
        for c in range(board_size):
            for i in range(1, board_size):
                if r + i < board_size: moves.append(((r, c), (r + i, c)))
                if r - i >= 0: moves.append(((r, c), (r - i, c)))
                if c + i < board_size: moves.append(((r, c), (r, c + i)))
                if c - i >= 0: moves.append(((r, c), (r, c - i)))
    
    unique_moves = sorted(list(set(moves)))
    action_to_idx = {move: i for i, move in enumerate(unique_moves)}
    idx_to_action = {i: move for i, move in enumerate(unique_moves)}
    return action_to_idx, idx_to_action

# --- GPU-Optimized Stratego Environment ---

class GPUOptimizedStrategoEnv:
    """GPU-optimized Stratego environment with recording capability."""
    def __init__(self, device, record_game=False, episode_num=None):
        self.device = device
        self.board_size = BOARD_SIZE
        self.record_game = record_game
        self.recorder = GameRecorder(episode_num) if record_game else None
        
        self.lakes = torch.tensor([(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)], device=device)
        self.directions = torch.tensor([(0, 1), (0, -1), (1, 0), (-1, 0)], device=device)
        
        self.board = torch.zeros((self.board_size, self.board_size), dtype=torch.int8, device=device)
        self.reset()

    def _precompute_positions(self):
        """Pre-compute all valid positions for faster move generation."""
        positions = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                # Skip lakes
                if not any((r == lake[0] and c == lake[1]) for lake in self.lakes):
                    positions.append((r, c))
        return torch.tensor(positions, device=self.device)

    def reset(self):
        """Resets the environment to a new game."""
        self.board.fill_(EMPTY_SQUARE)
        for r, c in self.lakes:
            self.board[r, c] = LAKE_SQUARE
        self._setup_board_gpu()
        
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.move_history = []

        if self.recorder:
            self.recorder.record_state(self.board, self.current_player)
            
        return self._get_state_tensor()

    def _setup_board_gpu(self):
        """GPU-optimized board setup with random piece placement."""
        pieces = [FLAG, SPY] + [BOMB]*6 + [MARSHAL] + [GENERAL] + [COLONEL]*2 + \
                 [MAJOR]*3 + [CAPTAIN]*4 + [LIEUTENANT]*4 + [SERGEANT]*4 + \
                 [MINER]*5 + [SCOUT]*8
        
        p1_pos = [(r, c) for r in range(6, 10) for c in range(10)]
        p2_pos = [(r, c) for r in range(0, 4) for c in range(10)]
        
        random.shuffle(p1_pos)
        random.shuffle(pieces)
        for i, piece in enumerate(pieces):
            r, c = p1_pos[i]
            self.board[r, c] = piece
        
        random.shuffle(p2_pos)
        random.shuffle(pieces)
        for i, piece in enumerate(pieces):
            r, c = p2_pos[i]
            self.board[r, c] = -piece

    def get_valid_moves_gpu(self):
        """Generates all valid moves for the current player."""
        moves = []
        player_pieces = torch.nonzero(self.board * self.current_player > 0)
        
        for r_from, c_from in player_pieces:
            r, c = r_from.item(), c_from.item()
            piece_rank = abs(self.board[r, c].item())

            if piece_rank in [BOMB, FLAG]: continue

            if piece_rank == SCOUT:
                for dr, dc in self.directions:
                    for i in range(1, self.board_size):
                        r_to, c_to = r + i * dr.item(), c + i * dc.item()
                        if not self._is_valid_target(r_to, c_to): break
                        moves.append(((r, c), (r_to, c_to)))
                        if self.board[r_to, c_to].item() != EMPTY_SQUARE: break
            else:
                for dr, dc in self.directions:
                    r_to, c_to = r + dr.item(), c + dc.item()
                    if self._is_valid_target(r_to, c_to):
                        moves.append(((r, c), (r_to, c_to)))
        return moves

    def _is_valid_target(self, r, c):
        """Checks if a target square is valid for a move."""
        if not (0 <= r < self.board_size and 0 <= c < self.board_size): return False
        target_val = self.board[r, c].item()
        return target_val != LAKE_SQUARE and (target_val * self.current_player) <= 0

    def step_gpu(self, action):
        """Executes a move and returns the new state, reward, and done flag."""
        if self.game_over:
            return self._get_state_tensor(), 0.0, True, {"winner": self.winner}
        
        (r_from, c_from), (r_to, c_to) = action
        moving_piece = self.board[r_from, c_from].item()
        target_piece = self.board[r_to, c_to].item()
        
        reward = -0.01 # Small penalty for each move to encourage finishing
        
        # Resolve battle or move
        if target_piece != EMPTY_SQUARE:
            winner_piece = self._resolve_battle(moving_piece, target_piece)
            if winner_piece == moving_piece:
                self.board[r_to, c_to] = moving_piece
                self.board[r_from, c_from] = EMPTY_SQUARE
                reward += 0.1 * abs(target_piece) # Reward for capture
                if abs(target_piece) == FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    reward += 1.0
            elif winner_piece == target_piece:
                self.board[r_from, c_from] = EMPTY_SQUARE
                reward -= 0.1 * abs(moving_piece) # Penalty for losing a piece
            else: # Draw
                self.board[r_from, c_from] = EMPTY_SQUARE
                self.board[r_to, c_to] = EMPTY_SQUARE
        else: # Move to empty square
            self.board[r_to, c_to] = moving_piece
            self.board[r_from, c_from] = EMPTY_SQUARE

        self.turn_count += 1
        self.move_history.append(action)
        self.current_player *= -1

        if not self.game_over:
            self._check_game_end()

        if self.recorder:
            self.recorder.record_state(self.board, self.current_player, action)
            
        info = {"winner": self.winner}
        return self._get_state_tensor(), reward, self.game_over, info
        
    def _resolve_battle(self, attacker, defender):
        """Resolves a battle based on Stratego rules."""
        atk_rank, def_rank = abs(attacker), abs(defender)
        if atk_rank == SPY and def_rank == MARSHAL: return attacker
        if atk_rank == MINER and def_rank == BOMB: return attacker
        if def_rank == BOMB: return defender
        if atk_rank > def_rank: return attacker
        if def_rank > atk_rank: return defender
        return None # Both removed

    def _check_game_end(self):
        """Checks for game-ending conditions."""
        if not any(abs(p.item()) == FLAG for p in self.board.flatten()):
             self.game_over = True # Should be caught by capture logic, but as a failsafe
             self.winner = -self.current_player

        if not self.get_valid_moves_gpu():
            self.game_over = True
            self.winner = -self.current_player # Player who cannot move loses
            
        if self.turn_count > 500: # Draw condition
            self.game_over = True
            self.winner = 0

    def _get_state_tensor(self):
        """Gets the state as a 4-channel tensor for the neural network."""
        state = torch.zeros(4, self.board_size, self.board_size, device=self.device)
        
        # Player and opponent piece planes
        player_board = self.board if self.current_player == 1 else -self.board
        state[0] = (player_board > 0) * player_board / 11.0 # Current player's pieces
        state[1] = (player_board < 0) * -player_board / 11.0 # Opponent's pieces (ranks are positive)
        
        # Turn and lake planes
        state[2].fill_(self.current_player)
        state[3] = (self.board == LAKE_SQUARE).float()
        
        return state.unsqueeze(0) # Add batch dimension

    def finalize_recording(self):
        """Finalizes and saves the timelapse video."""
        if self.recorder:
            self.recorder.create_timelapse(winner=self.winner)

# --- Neural Network ---

class ResidualBlock(nn.Module):
    """A residual block for the CNN."""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class StrategoNet(nn.Module):
    """The neural network for the Stratego agent."""
    def __init__(self, num_res_blocks, num_channels, num_actions):
        super().__init__()
        self.conv_in = nn.Conv2d(4, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, num_actions)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

# --- GPU-Accelerated MCTS ---

class GPUAcceleratedMCTS:
    """GPU-accelerated Monte Carlo Tree Search."""
    def __init__(self, model, device, num_simulations=50):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = 1.4

    def search(self, env):
        """Performs MCTS to find the best action."""
        root = MCTSNode()
        
        for _ in range(self.num_simulations):
            node = root
            sim_env = deepcopy(env) # Create a copy for simulation
            
            # 1. Selection
            while node.is_expanded:
                if not node.children: break
                node = node.select_child()
                sim_env.step_gpu(node.action)

            # 2. Expansion & Evaluation
            if not sim_env.game_over:
                with torch.no_grad():
                    state_tensor = sim_env._get_state_tensor()
                    policy, value = self.model(state_tensor)
                
                value = value.item()
                valid_moves = sim_env.get_valid_moves_gpu()
                
                if valid_moves:
                    policy = torch.exp(policy).squeeze(0)
                    action_probs = []
                    for move in valid_moves:
                        if move in ACTION_TO_IDX:
                           action_probs.append((move, policy[ACTION_TO_IDX[move]].item()))
                    
                    if action_probs:
                        node.expand(action_probs)
            else:
                 # Terminal node value
                 if sim_env.winner == sim_env.current_player: value = 1.0
                 elif sim_env.winner == -sim_env.current_player: value = -1.0
                 else: value = 0.0

            # 3. Backpropagation
            node.backup(value)
        
        if not root.children: return None
        
        # Select action based on visit counts
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        
        # For training data, create a policy vector based on visit counts
        policy_target = torch.zeros(len(ACTION_TO_IDX), device=self.device)
        for action, child in root.children.items():
            if action in ACTION_TO_IDX:
                policy_target[ACTION_TO_IDX[action]] = child.visits
        policy_target /= root.visits
        
        q_value = root.total_value / root.visits

        return best_action, policy_target, q_value

# --- Agent and Training ---

class ReplayBuffer:
    """A simple replay buffer."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class StrategoAgent:
    """The agent that learns and plays the game."""
    def __init__(self, model, device, lr=0.001, replay_buffer_size=10000):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.mcts = GPUAcceleratedMCTS(model, device, num_simulations=50)

    def choose_action(self, env):
        """Chooses an action using MCTS."""
        return self.mcts.search(env)

    def train_step(self, batch_size):
        """Performs one training step."""
        if len(self.replay_buffer) < batch_size: return 0.0
        
        samples = self.replay_buffer.sample(batch_size)
        states, policies, values = zip(*samples)
        
        states = torch.cat(states).to(self.device)
        policies = torch.stack(policies).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        pred_policies, pred_values = self.model(states)
        
        policy_loss = F.kl_div(pred_policies, policies, reduction='batchmean')
        value_loss = F.mse_loss(pred_values, values)
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Hyperparameters ---
    NUM_EPISODES = 1000
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 20000
    TRAIN_AFTER_EPISODES = 10
    MCTS_SIMULATIONS = 40
    NUM_RES_BLOCKS = 5
    NUM_CHANNELS = 128
    
    # --- Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    ACTION_TO_IDX, IDX_TO_ACTION = generate_all_possible_moves(BOARD_SIZE)
    NUM_ACTIONS = len(ACTION_TO_IDX)

    model = StrategoNet(NUM_RES_BLOCKS, NUM_CHANNELS, NUM_ACTIONS).to(DEVICE)
    agent = StrategoAgent(model, DEVICE, lr=LEARNING_RATE, replay_buffer_size=REPLAY_BUFFER_SIZE)
    agent.mcts.num_simulations = MCTS_SIMULATIONS
    
    plotter = PeriodicResultsPlotter(plot_frequency=50)
    
    # --- Training History ---
    history = {
        'wins': [],
        'loss': [],
        'game_lengths': [],
        'q_values': [],
        'gpu_mem': []
    }

    # --- Training Loop ---
    for episode in range(NUM_EPISODES):
        # Decide if this episode should be recorded
        record_this_game = (episode % 100 == 0) or (episode == NUM_EPISODES - 1)
        
        env = GPUOptimizedStrategoEnv(DEVICE, record_game=record_this_game, episode_num=episode)
        state = env.reset()
        done = False
        
        game_data = []
        episode_q_values = []

        start_time = time.time()
        while not done:
            action_result = agent.choose_action(env)
            if action_result is None: # No valid moves found
                break
            
            action, policy_target, q_value = action_result
            episode_q_values.append(q_value)

            # Store experience for replay buffer
            game_data.append((env._get_state_tensor(), policy_target))

            state, reward, done, info = env.step_gpu(action)

        # Game finished, process results
        winner = info.get("winner", 0)
        
        # Add experiences to replay buffer with final outcome
        for state_tensor, policy in game_data:
            value = 0
            if winner == state_tensor[0, 2, 0, 0].item(): value = 1.0 # If the player to move won
            elif winner == -state_tensor[0, 2, 0, 0].item(): value = -1.0 # If the player to move lost
            agent.replay_buffer.push(state_tensor, policy, value)

        # Update history
        history['wins'].append(winner)
        history['game_lengths'].append(env.turn_count)
        if episode_q_values:
            history['q_values'].append(np.mean(episode_q_values))
        if DEVICE.type == 'cuda':
             history['gpu_mem'].append(torch.cuda.memory_allocated(DEVICE) / 1024**2)

        # Logging
        duration = time.time() - start_time
        print(f"Episode {episode}/{NUM_EPISODES} | Winner: {winner} | Turns: {env.turn_count} | Duration: {duration:.2f}s")
        
        # Finalize recording if it was enabled
        if record_this_game:
            env.finalize_recording()
            
        # Train the model periodically
        if episode > 0 and episode % TRAIN_AFTER_EPISODES == 0:
            print("--- Training Step ---")
            loss = agent.train_step(BATCH_SIZE)
            history['loss'].append(loss)
            print(f"Loss: {loss:.4f} | Buffer Size: {len(agent.replay_buffer)}")

        # Create periodic plot
        if plotter.should_plot(episode):
            plotter.create_periodic_plot(episode, history)

    print("\nTraining finished!")
    # Save final model
    torch.save(model.state_dict(), "stratego_final_model.pth")
    print("✅ Final model saved.")