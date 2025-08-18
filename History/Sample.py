# Stratego with Multi-Agent Deep Q-Learning
# This script implements the game of Stratego and trains two AI agents
# to play against each other using Deep Q-Networks (DQN) with PyTorch.
# Includes matplotlib for visualizing training progress.
# VERSION 3: Optimized tensor conversion to address UserWarning.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from copy import deepcopy
import matplotlib.pyplot as plt

# --- Configuration ---
BOARD_SIZE = 10
NUM_PIECES = 40 # Pieces per player
HIDDEN_PIECE = -1
EMPTY_SQUARE = 0
LAKE_SQUARE = -2

# Piece Ranks (Value represents strength)
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

# --- Game Logic and Environment ---

class StrategoEnv:
    """
    Stratego Game Environment.
    Manages game state, rules, and interactions for RL agents.
    """
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.lakes = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]
        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.board = np.full((self.board_size, self.board_size), EMPTY_SQUARE, dtype=int)
        for r, c in self.lakes:
            self.board[r, c] = LAKE_SQUARE

        # Player pieces are positive, opponent pieces are negative in internal representation
        self.pieces = {1: self._get_initial_pieces(), -1: self._get_initial_pieces()}
        self._setup_board()

        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self._get_state()

    def _get_initial_pieces(self):
        """Returns the standard set of Stratego pieces."""
        return {
            FLAG: 1, BOMB: 6, MARSHAL: 1, GENERAL: 1, COLONEL: 2, MAJOR: 3,
            CAPTAIN: 4, LIEUTENANT: 4, SERGEANT: 4, MINER: 5, SCOUT: 8, SPY: 1
        }

    def _setup_board(self):
        """Places pieces on the board for a new game."""
        # For this simulation, we use a fixed, simple setup.
        # A real implementation would involve a setup phase.
        for player in [1, -1]:
            pieces_to_place = []
            for rank, count in self.pieces[player].items():
                pieces_to_place.extend([rank] * count)
            
            random.shuffle(pieces_to_place)

            if player == 1: # Player 1 (bottom)
                rows = range(6, 10)
            else: # Player -1 (top)
                rows = range(0, 4)

            idx = 0
            for r in rows:
                for c in range(self.board_size):
                    if idx < len(pieces_to_place):
                        self.board[r, c] = pieces_to_place[idx] * player
                        idx += 1

    def _get_state(self):
        """
        Generates the state representation for the current player.
        The state is a 3-channel numpy array.
        """
        player = self.current_player
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)

        for r in range(self.board_size):
            for c in range(self.board_size):
                val = self.board[r, c]
                if val * player > 0:  # Player's own piece
                    state[0, r, c] = val * player
                elif val * player < 0:  # Opponent's piece
                    # For simplicity, we assume perfect information for this DQN example.
                    state[1, r, c] = abs(val)
                elif val == LAKE_SQUARE:
                    state[2, r, c] = 1.0
        return state

    def get_valid_moves(self):
        """Returns a list of all valid moves for the current player."""
        moves = []
        player = self.current_player
        for r_from in range(self.board_size):
            for c_from in range(self.board_size):
                if self.board[r_from, c_from] * player > 0:
                    piece_rank = abs(self.board[r_from, c_from])
                    
                    if piece_rank == BOMB or piece_rank == FLAG:
                        continue

                    # Standard one-step moves
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        r_to, c_to = r_from + dr, c_from + dc
                        if self._is_valid_target(r_to, c_to, player):
                            moves.append(((r_from, c_from), (r_to, c_to)))

                    # Scout's special multi-step move
                    if piece_rank == SCOUT:
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            for i in range(2, self.board_size):
                                r_to, c_to = r_from + i * dr, c_from + i * dc
                                if self._is_valid_target(r_to, c_to, player):
                                    moves.append(((r_from, c_from), (r_to, c_to)))
                                    if self.board[r_to, c_to] != EMPTY_SQUARE:
                                        break 
                                else:
                                    break
        return moves
    
    def _is_valid_target(self, r, c, player):
        """Checks if a target square (r, c) is a valid destination."""
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
        if self.board[r, c] == LAKE_SQUARE:
            return False
        if self.board[r, c] * player > 0:
            return False
        return True

    def step(self, action):
        """
        Executes a move and returns the new state, reward, and game over status.
        """
        if self.game_over:
            return self._get_state(), 0, True, self.winner

        (r_from, c_from), (r_to, c_to) = action
        player = self.current_player
        moving_piece_val = self.board[r_from, c_from]
        target_piece_val = self.board[r_to, c_to]
        
        moving_rank = abs(moving_piece_val)
        target_rank = abs(target_piece_val)

        reward = -0.01

        if target_piece_val != EMPTY_SQUARE:
            if moving_rank == SPY and target_rank == MARSHAL:
                winner = moving_piece_val
            elif moving_rank == MINER and target_rank == BOMB:
                winner = moving_piece_val
            elif moving_rank > target_rank:
                winner = moving_piece_val
            elif target_rank > moving_rank:
                winner = target_piece_val
            else:
                winner = None

            if winner == moving_piece_val:
                self.board[r_to, c_to] = moving_piece_val
                self.board[r_from, c_from] = EMPTY_SQUARE
                reward += 0.1 * target_rank
                if target_rank == FLAG:
                    self.game_over = True
                    self.winner = player
                    reward += 10.0
            elif winner == target_piece_val:
                self.board[r_from, c_from] = EMPTY_SQUARE
                reward -= 0.1 * moving_rank
            else:
                self.board[r_to, c_to] = EMPTY_SQUARE
                self.board[r_from, c_from] = EMPTY_SQUARE
        else:
            self.board[r_to, c_to] = moving_piece_val
            self.board[r_from, c_from] = EMPTY_SQUARE

        if not self.game_over:
            opponent = -player
            if not self._get_valid_moves_for_player(opponent):
                self.game_over = True
                self.winner = player
                reward += 10.0

        self.move_history.append(action)
        self.current_player = -self.current_player
        
        return self._get_state(), reward, self.game_over, {"winner": self.winner}

    def _get_valid_moves_for_player(self, player):
        original_player = self.current_player
        self.current_player = player
        moves = self.get_valid_moves()
        self.current_player = original_player
        return moves

    def render(self):
        print("\n" + "="*30)
        print(f"Turn: Player {'1 (v)' if self.current_player == 1 else '2 (^)'}")
        for r in range(self.board_size):
            line = ""
            for c in range(self.board_size):
                val = self.board[r, c]
                player_owner = np.sign(val)
                rank = abs(val)
                char = PIECE_NAMES[rank]
                
                if player_owner == 1:
                    line += f" \033[94m{char}\033[0m "
                elif player_owner == -1:
                    line += f" \033[91m{char}\033[0m "
                else:
                    line += f" {char} "
            print(line)
        print("="*30)

# --- Deep Q-Network (DQN) ---

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        linear_input_size = 64 * h * w
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# --- DQN Agent ---

# Storing raw numpy arrays in memory is more efficient
ReplayMemory = collections.namedtuple('ReplayMemory',
                                      ('state', 'action_index', 'next_state', 'reward'))

class DQNAgent:
    def __init__(self, player_id, n_actions, device):
        self.player_id = player_id
        self.device = device
        self.n_actions = n_actions
        
        self.policy_net = DQN(BOARD_SIZE, BOARD_SIZE, n_actions).to(device)
        self.target_net = DQN(BOARD_SIZE, BOARD_SIZE, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = collections.deque(maxlen=10000)

        self.steps_done = 0
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 5000

    def select_action(self, state, valid_moves, all_possible_moves):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Convert single state to tensor for model evaluation
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)[0]
                
                mask = torch.full(q_values.shape, -float('inf'), device=self.device)
                valid_indices = [all_possible_moves.index(move) for move in valid_moves]
                mask[valid_indices] = 0
                
                masked_q_values = q_values + mask
                
                max_q_value = masked_q_values.max().item()
                action_index = masked_q_values.argmax().item()
                return all_possible_moves[action_index], action_index, max_q_value
        else:
            action = random.choice(valid_moves)
            action_index = all_possible_moves.index(action)
            return action, action_index, None

    def push_memory(self, state, action_index, next_state, reward):
        """Saves a transition with raw numpy arrays to the replay memory."""
        # Store raw numpy arrays and primitives, not tensors
        self.memory.append(ReplayMemory(state, action_index, next_state, reward))

    def optimize_model(self, batch_size=128, gamma=0.99):
        if len(self.memory) < batch_size:
            return None
        
        transitions = random.sample(self.memory, batch_size)
        batch = ReplayMemory(*zip(*transitions))

        # **OPTIMIZATION**: Convert batch of numpy arrays to a single tensor
        state_batch = torch.from_numpy(np.array(batch.state)).to(self.device)
        action_batch = torch.tensor(batch.action_index, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=self.device, dtype=torch.bool)
        
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        # **OPTIMIZATION**: Also convert the next_states batch efficiently
        non_final_next_states = torch.from_numpy(np.array(non_final_next_states_list)).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- Training Loop & Plotting ---

def generate_all_possible_moves(board_size):
    all_moves = []
    for r_from in range(board_size):
        for c_from in range(board_size):
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                r_to, c_to = r_from + dr, c_from + dc
                if 0 <= r_to < board_size and 0 <= c_to < board_size:
                    all_moves.append(((r_from, c_from), (r_to, c_to)))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for i in range(2, board_size):
                    r_to, c_to = r_from + i * dr, c_from + i * dc
                    if 0 <= r_to < board_size and 0 <= c_to < board_size:
                        all_moves.append(((r_from, c_from), (r_to, c_to)))
    return all_moves

def plot_progress(win_history, q_history, loss_history):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    win_p1 = np.array([1 if w == 1 else 0 for w in win_history])
    win_p2 = np.array([1 if w == -1 else 0 for w in win_history])
    moving_avg_p1 = np.convolve(win_p1, np.ones(50)/50, mode='valid')
    moving_avg_p2 = np.convolve(win_p2, np.ones(50)/50, mode='valid')
    plt.plot(moving_avg_p1, label='Player 1 Wins (50-ep MA)', color='blue')
    plt.plot(moving_avg_p2, label='Player 2 Wins (50-ep MA)', color='red')
    plt.title('Win Rate (Moving Average)')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(q_history['p1'], label='Player 1 Avg Max Q', color='blue', alpha=0.7)
    plt.plot(q_history['p2'], label='Player 2 Avg Max Q', color='red', alpha=0.7)
    plt.title('Average Max Q-Value per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(loss_history['p1'], label='Player 1 Loss', color='blue', alpha=0.7)
    plt.plot(loss_history['p2'], label='Player 2 Loss', color='red', alpha=0.7)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = StrategoEnv()
    all_possible_moves = generate_all_possible_moves(BOARD_SIZE)
    n_actions = len(all_possible_moves)
    print(f"Total possible move actions: {n_actions}")

    agent1 = DQNAgent(player_id=1, n_actions=n_actions, device=device)
    agent2 = DQNAgent(player_id=-1, n_actions=n_actions, device=device)
    agents = {1: agent1, -1: agent2}

    num_episodes = 500
    target_update_frequency = 10

    win_history = []
    q_history = {'p1': [], 'p2': []}
    loss_history = {'p1': [], 'p2': []}

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        
        last_states = {1: None, -1: None}
        last_action_indices = {1: None, -1: None}
        
        episode_q_vals = {'p1': [], 'p2': []}
        episode_losses = {'p1': [], 'p2': []}

        while not done:
            player = env.current_player
            current_agent = agents[player]
            
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                done = True
                env.winner = -player
                if last_states[player] is not None:
                     agents[player].push_memory(last_states[player], last_action_indices[player], state, -10.0)
                if last_states[-player] is not None:
                     agents[-player].push_memory(last_states[-player], last_action_indices[-player], state, 10.0)
                continue

            action, action_index, max_q = current_agent.select_action(state, valid_moves, all_possible_moves)
            if max_q is not None:
                episode_q_vals['p1' if player == 1 else 'p2'].append(max_q)

            last_states[player] = state
            last_action_indices[player] = action_index

            next_state, reward, done, info = env.step(action)
            
            opponent = -player
            if last_states[opponent] is not None:
                agents[opponent].push_memory(last_states[opponent], last_action_indices[opponent], state, -reward)
                last_states[opponent] = None

            if done:
                current_agent.push_memory(state, action_index, None, reward)
            
            state = next_state

            loss1 = agent1.optimize_model()
            if loss1: episode_losses['p1'].append(loss1)
            loss2 = agent2.optimize_model()
            if loss2: episode_losses['p2'].append(loss2)

        win_history.append(env.winner)
        
        avg_q1 = np.mean(episode_q_vals['p1']) if episode_q_vals['p1'] else 0
        avg_q2 = np.mean(episode_q_vals['p2']) if episode_q_vals['p2'] else 0
        q_history['p1'].append(avg_q1)
        q_history['p2'].append(avg_q2)

        avg_loss1 = np.mean(episode_losses['p1']) if episode_losses['p1'] else 0
        avg_loss2 = np.mean(episode_losses['p2']) if episode_losses['p2'] else 0
        loss_history['p1'].append(avg_loss1)
        loss_history['p2'].append(avg_loss2)

        print(f"Episode {i_episode+1}/{num_episodes} finished. Winner: Player {'1' if env.winner == 1 else '2' if env.winner == -1 else 'Draw'}")

        if (i_episode + 1) % target_update_frequency == 0:
            print("--- Updating Target Networks ---")
            agent1.update_target_net()
            agent2.update_target_net()
            
    print("Training finished.")
    plot_progress(win_history, q_history, loss_history)
    
    # torch.save(agent1.policy_net.state_dict(), 'agent1_policy_net.pth')
    # torch.save(agent2.policy_net.state_dict(), 'agent2_policy_net.pth')

if __name__ == '__main__':
    main()
