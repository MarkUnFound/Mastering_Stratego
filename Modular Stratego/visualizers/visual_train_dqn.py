"""
Visual Training Script for Rainbow DQN Agents in Stratego
INTERACTIVE MODE: Single-lane, visualizations enabled.

Features:
- Real-time visualization of Board, Q-Values, and Gradients.
- Interactive controls (Toggle AAREN, Pause, etc.)
- Visual Curriculum Transitions
"""

# Force interactive backend
import matplotlib
import sys
import os

# Try to set backend to TkAgg (Standard Interactive)
try:
    matplotlib.use('TkAgg') 
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import numpy as np
import random
import time
from collections import deque

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drqn_agent import DQNAgent
from environment import StrategoEnvironment
from game_state import GameState
from piece import PieceType, PIECE_RANKS, PIECE_NAMES
from board import LAKE_SQUARE
from training_config import *
from opponents import RandomAgent, OpponentPool, RandomSetupAgent, GreedyAgent
from heuristic_setup import HeuristicSetupAgent
from league import LeagueManager
from distributional_reward import StrategoRewardConfig
from policy_search import PolicyRefinedSearch, SearchConfig
from curriculum import CurriculumManager

# Helper to capture gradients
class GradientCatcher:
    def __init__(self):
        self.grads = {}
        
    def hook(self, module, grad_in, grad_out):
        if grad_out and len(grad_out) > 0:
            self.grads['input_grad'] = grad_out[0].detach().cpu()

class VisualDQNAgent(DQNAgent):
    """
    Subclass of DQNAgent that exposes internal state for visualization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_q_values = None
        self.last_top_moves = []
        self.last_uncertainty_map = None
        self.gradient_catcher = GradientCatcher()
        self.last_search_info = None  # Search visualization data
        self.search_engine = None  # Will be set externally
        
        # Register hook on the first convolutional layer to catch input gradients (Saliency)
        self._register_hooks()

    def _register_hooks(self):
        # Hook into the first convolution layer of the Q-network
        first_layer = self.q_network.conv_in
        # Using register_full_backward_hook is safer for modern PyTorch
        self._hook_handle = first_layer.register_full_backward_hook(self.gradient_catcher.hook)

    def reset(self):
        super().reset()
        if hasattr(self, '_hook_handle'):
            self._hook_handle.remove()
        self._register_hooks()

    def get_last_visual_data(self):
        return {
            'q_values': self.last_q_values,
            'top_moves': self.last_top_moves,
            'uncertainty': self.last_uncertainty_map,
            'gradients': self.gradient_catcher.grads.get('input_grad'),
            'search_info': self.last_search_info
        }

    def act_visual(self, state, valid_moves, game_state=None, use_search=False):
        """
        Same as act(), but stores the Q-values for visualization.
        Optionally uses test-time search if enabled.
        """
        # Respect the toggle: if use_pbs is False, pass None to force internal padding
        history_instance = self.history if self.use_pbs else None
        
        state_tensor = self.get_state_representation(state, pbs_instance=history_instance)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.q_network.eval()
        with torch.no_grad():
            log_probs = self.q_network(state_tensor)
            probs = log_probs.exp()
            expected_q_values = (probs * self.support).sum(dim=2) # (1, actions)
            base_q_values = expected_q_values.squeeze(0) # (actions)
            
        self.q_network.train()
        
        # --- Visualization Capture ---
        q_map = np.full((10, 10), -np.inf)
        
        # Uncertainty is now implicitly handled via AAREN embeddings
        uncertainty_map = {}
            
        valid_q_values = []
        best_q = -float('inf')

        for move in valid_moves:
            action_idx = self._move_to_action_index(move)
            q_val = base_q_values[action_idx].item()
            
            uncertainty = self.get_move_uncertainty(move, uncertainty_map)
            exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
            final_q = q_val + exploration_bonus
            
            valid_q_values.append(final_q)
            
            (r1, c1), (r2, c2) = move
            # Map Q-value to DESTINATION square for heatmap
            if final_q > q_map[r2, c2]:
                q_map[r2, c2] = final_q
        
                q_map[r2, c2] = final_q
        
        # Sort top moves
        top_indices = np.argsort(valid_q_values)[::-1][:10]
        self.last_top_moves = []
        for idx in top_indices:
            move = valid_moves[idx]
            q = valid_q_values[idx]
            self.last_top_moves.append((move, q))

        self.last_q_values = q_map
        self.last_uncertainty_map = uncertainty_map
        
        if not valid_q_values:
            return None
        
        # Test-Time Search (Optional)
        self.last_search_info = None
        if use_search and self.search_engine is not None and len(valid_moves) >= 10:
            def step_fn(gs, move):
                # Simplified step function for search
                return gs, 0.0, False, {}
            def get_valid_moves_fn(gs):
                return valid_moves
            
            search_move, search_info = self.search_engine.search(
                state_tensor, valid_moves, get_valid_moves_fn, step_fn, 
                self.player_id, game_state
            )
            self.last_search_info = search_info
            if search_move:
                return search_move
            
        best_move_idx = np.argmax(valid_q_values)
        return valid_moves[best_move_idx]


def visual_train():
    print("Starting Interactive Visual Training...")
    
    # --- Backend Debugging ---
    backend = matplotlib.get_backend()
    print(f"  Matplotlib Backend: {backend}")
    if 'Agg' in backend and 'Tk' not in backend and 'Qt' not in backend:
        print("  WARNING: Non-interactive backend detected! Window likely won't show.")
        print("   Checking if Tkinter is available...")
        try:
            import tkinter
            print("    Tkinter is installed.")
        except ImportError:
            print("    Tkinter NOT found. Please install python-tk or check your environment.")
    
    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Environment
    env = StrategoEnvironment(device=device)
    
    # Agents
    print("Initializing Agents...")
    agent1 = VisualDQNAgent(player_id=1, device=device, lr=LEARNING_RATE, buffer_size=10000)
    
    # Opponent Infrastructure (Aligned with train_dqn.py)
    master_reward_config = StrategoRewardConfig.from_training_config()
    model_save_path = "dqn_models" # Default path
    league_dir = os.path.join(model_save_path, "league")
    league_manager = LeagueManager(league_dir=league_dir, max_agents=LEAGUE_MAX_AGENTS)
    
    opponent_pool = OpponentPool(
        league_manager=league_manager,
        device=device,
        league_prob=OPPONENT_LEAGUE_PROB,
        random_prob=OPPONENT_RANDOM_PROB,
        greedy_prob=OPPONENT_GREEDY_PROB,
        self_prob=OPPONENT_SELF_PROB, 
        config=master_reward_config
    )
    
    # Specific Agents
    dqn_agent2 = DQNAgent(player_id=-1, device=device, lr=LEARNING_RATE, buffer_size=10000, use_pbs=False)
    random_agent = RandomAgent()
    greedy_agent = GreedyAgent(device=device, player_id=-1, config=master_reward_config)
    
    # Setup Agents
    setup_agent1 = HeuristicSetupAgent(player_id=1, device=device)
    setup_agent2 = HeuristicSetupAgent(player_id=-1, device=device)
    random_setup_agent = RandomSetupAgent(player_id=-1)
    
    curriculum = CurriculumManager(start_phase=CURRICULUM_START_PHASE)
    print(f"Curriculum initialized: Phase {curriculum.current_phase.value}")
    
    # Test-Time Search Engine (for visualization only)
    search_config = SearchConfig(
        enabled=True,
        search_depth=2,
        top_k_moves=5,
        search_budget=50
    )
    agent1.search_engine = PolicyRefinedSearch(
        agent1.q_network, search_config, device=str(device)
    )
    print("[OK] Test-Time Search engine initialized for visualization")
    
    agent2_placeholder = rainbow_agent2 # Default reference
    
    # --- Visualization Setup ---
    print(" Initializing Plot Window...")
    plt.ion() # Interactive Mode ON
    
    try:
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Interactive Stratego Training (Phase {curriculum.current_phase.value})", fontsize=16)
        
        # Force Window Show
        plt.show(block=False)
        plt.pause(0.5) # Give time to render
        print(" Window initialized.")
    except Exception as e:
        print(f" Failed to initialize window: {e}")
        return

    gs = GridSpec(2, 5, figure=fig)
    
    # Plots
    ax_board = fig.add_subplot(gs[0:2, 0:2])
    ax_board.set_title("Live Game Board")
    
    ax_qval = fig.add_subplot(gs[0, 2])
    ax_qval.set_title("Movement Intent (Heatmap)")
    
    ax_top_moves = fig.add_subplot(gs[0, 3])
    ax_top_moves.set_title("Top 10 Candidate Moves")
    ax_top_moves.axis('off')
    
    ax_backprop = fig.add_subplot(gs[1, 2:])
    ax_backprop.set_title("Network Attention (Backprop Saliency)")
    
    ax_info = fig.add_subplot(gs[:, 4])
    ax_info.axis('off')
    
    # Shared State
    state_vars = {
        'paused': False,
        'step_by_step': False,
        'run_speed': 0.1,
        'aaren_active': True,
        'search_active': False,  # Test-time search toggle
        'manual_step_trigger': False
    }
    
    agent1.use_pbs = True
    
    def update_info_text():
        text = (
            f"Phase: {curriculum.current_phase.name} ({curriculum.current_phase.value})\n"
            f"Step Speed: {state_vars['run_speed']:.2f}s\n"
            f"AAREN Active: {state_vars['aaren_active']}\n"
            f"Search Active: {state_vars['search_active']}\n"
            f"Paused: {state_vars['paused']}\n\n"
            "CONTROLS:\n"
            "[SPACE]: Pause/Resume\n"
            "[RIGHT]: Step (if paused)\n"
            "[UP/DOWN]: Speed +/- \n"
            "[A]: Toggle AAREN\n"
            "[S]: Toggle Search\n"
            "[C]: Force Curriculum Check"
        )
        ax_info.clear()
        ax_info.axis('off')
        ax_info.text(0, 0.9, text, fontsize=12, va='top', fontfamily='monospace')

    def on_key(event):
        if event.key == ' ':
            state_vars['paused'] = not state_vars['paused']
        elif event.key == 'right':
            state_vars['manual_step_trigger'] = True
        elif event.key == 'up':
            state_vars['run_speed'] = max(0.01, state_vars['run_speed'] / 2)
        elif event.key == 'down':
            state_vars['run_speed'] = min(2.0, state_vars['run_speed'] * 2)
        elif event.key == 'a':
            state_vars['aaren_active'] = not state_vars['aaren_active']
            agent1.use_pbs = state_vars['aaren_active']
            print(f"AAREN Toggled: {state_vars['aaren_active']}")
        elif event.key == 's':
            state_vars['search_active'] = not state_vars['search_active']
            print(f" Search Toggled: {state_vars['search_active']}")
        elif event.key == 'c':
             # Force verify curriculum
            curriculum.update_metrics({'winner': 1, 'pbs_accuracy': 0.8}) 
            fig.suptitle(f"Interactive Stratego Training (Phase {curriculum.current_phase.value}: {curriculum.current_phase.name})", fontsize=16)

        update_info_text()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_info_text()
    
    losses = []
    episode = 0
    total_steps = 0
    
    # --- Loop ---
    try:
        while True:
            # 1. Determine Opponent
            opponent_type = "self"
            opponent_uses_history = True
            current_opponent = rainbow_agent2
            
            if curriculum:
                opponent_dist = curriculum.get_opponent_distribution()
                r = random.random()
                cumulative = 0.0
                for op_type, prob in opponent_dist.items():
                    cumulative += prob
                    if r < cumulative:
                        opponent_type = op_type
                        break
                
                # Configure opponent (Simplified selection)
                if opponent_type == "random":
                    current_opponent = random_agent
                    opponent_uses_history = False
                elif opponent_type == "greedy":
                    current_opponent = greedy_agent
                    opponent_uses_history = False
                elif opponent_type == "league":
                    path = league_manager.get_opponent()
                    if path:
                        rainbow_agent2.load_model(path)
                        current_opponent = rainbow_agent2
                        opponent_uses_history = True
                    else:
                        opponent_type = "self" # Fallback
                        current_opponent = rainbow_agent2
                        # Sync weights for self-play
                        rainbow_agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                else: # self
                    current_opponent = rainbow_agent2
                    # Sync weights for self-play
                    rainbow_agent2.q_network.load_state_dict(agent1.q_network.state_dict())
            
            # 2. Setup Board
            # Generate pieces (using env helper if available, or manually)
            # Revisit: env._generate_pieces is protected. We can access or copy.
            # Using env call style from train_dqn implies we should respect environment logic
            
            p1_pieces = env._generate_pieces()
            p1_pos = env.get_valid_placement_positions(1)
            p1_place = setup_agent1.place_pieces(p1_pieces, p1_pos)
            
            p2_pieces = env._generate_pieces()
            p2_pos = env.get_valid_placement_positions(-1)
            
            p2_place = setup_agent2.place_pieces(p2_pieces, p2_pos)
                
            # Random starting player swap (50%)
            if random.random() < 0.5:
                p1_place, p2_place = p2_place, p1_place
                
            # Reset Environment with placements
            state = env.reset(p1_placement=p1_place, p2_placement=p2_place)
            valid_moves = env.get_valid_moves()
            
            done = False
            episode_reward1 = 0
            steps = 0
            
            # Ensure AAREN history state is reset
            if agent1.history: agent1.reset_history()
            if opponent_uses_history and hasattr(current_opponent, 'reset_history'): 
                current_opponent.reset_history()
            
            while not done:
                # Pause Logic
                if state_vars['paused']:
                    if state_vars['manual_step_trigger']:
                        state_vars['manual_step_trigger'] = False
                        # Proceed one step
                    else:
                        # Required to keep GUI alive during pause
                        fig.canvas.flush_events() 
                        time.sleep(0.1)
                        continue
                
                # 1. Visualization
                # 1. Visualization
                # Distinguish Ground Truth vs Observation for proper "Reveal" mechanics
                # ground_truth: env.board.actual_board (Always exists)
                # observation: env.board.get_visible_board(1) (What Agent 1 sees)
                
                ground_truth = env.board.actual_board
                observation = None
                
                if state_vars['aaren_active']:
                    # Fog Mode: Use observation to filter visibility
                    observation = env.board.get_visible_board(1)
                else:
                    # Cheat Mode: observation is None -> Everything revealed
                    observation = None

                ax_board.clear()
                visualize_board(ax_board, ground_truth, valid_moves, 
                              last_move=action if 'action' in locals() and action else None, 
                              observed_board=observation)
                
                # 2. Agent 1 Act
                if env.current_player == 1:
                    action = agent1.act_visual(state, valid_moves, game_state=state, 
                                              use_search=state_vars['search_active'])
                    
                    # Update Q-Map
                    data = agent1.get_last_visual_data()
                    if data['q_values'] is not None:
                        ax_qval.clear()
                        im = ax_qval.imshow(data['q_values'], cmap='viridis', interpolation='nearest', origin='upper')
                        ax_qval.set_title("Movement Intent (Heatmap)")
                        # Center labels on cells
                        ax_qval.set_xticks(range(10))
                        ax_qval.set_yticks(range(10))
                        # Minor ticks for grid lines
                        ax_qval.set_xticks(np.arange(-0.5, 10.5, 1), minor=True)
                        ax_qval.set_yticks(np.arange(-0.5, 10.5, 1), minor=True)
                        ax_qval.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

                    # Update Top Moves List (with search indicator)
                    if data['top_moves']:
                         visualize_top_moves(ax_top_moves, data['top_moves'], 
                                           search_info=data.get('search_info'))
                    
                    # Update Backprop (If available from training step)
                    grad_data = data['gradients']
                    if grad_data is not None:
                        # grad_data: (Batch, Channels, 10, 10)
                        saliency = grad_data.abs().mean(dim=0).mean(dim=0).numpy()
                        ax_backprop.clear()
                        ax_backprop.imshow(saliency, cmap='hot', origin='upper')
                        ax_backprop.set_title("Backprop Saliency")
                        # Center labels on cells
                        ax_backprop.set_xticks(range(10))
                        ax_backprop.set_yticks(range(10))
                        # Minor ticks for grid lines
                        ax_backprop.set_xticks(np.arange(-0.5, 10.5, 1), minor=True)
                        ax_backprop.set_yticks(np.arange(-0.5, 10.5, 1), minor=True)
                        ax_backprop.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
                    else:
                        ax_backprop.clear()
                        ax_backprop.text(0.5, 0.5, "No Backprop Data Yet", ha='center')
                        ax_backprop.set_xticks([])
                        ax_backprop.set_yticks([])

                else:
                    action = current_opponent.act(state, valid_moves)
                    
                    # Update AAREN history for Agent 1 based on opponent move (inference)
                    if agent1.history and agent1.use_pbs:
                        if action:
                            agent1.update_history_batch([action], [state], acting_player=-1)
                
                # 3. Step
                next_state, reward, done, info = env.step(action)
                next_valid_moves = env.get_valid_moves() if not done else []
                
                # 4. Agent 1 Learn
                if env.current_player == 1:
                    episode_reward1 += reward
                    agent1.remember(state, action, reward, next_state, done)
                    
                    # Train step (Backprop visualization)
                    if len(agent1.memory) > BATCH_SIZE and total_steps % 4 == 0:
                        loss = agent1.replay(episode)
                        if loss:
                            losses.append(loss)
                            
                            grad_data = agent1.get_last_visual_data()['gradients']
                            # Already visualized in real-time block above if available? 
                            # Actually, backprop happens HERE. So we need to update visual here or next frame.
                            # Let's update it here for responsiveness.
                            if grad_data is not None:
                                saliency = grad_data.abs().mean(dim=0).mean(dim=0).numpy()
                                ax_backprop.clear()
                                ax_backprop.imshow(saliency, cmap='hot', origin='upper')
                                ax_backprop.set_title("Backprop Saliency")
                                # Center labels on cells
                                ax_backprop.set_xticks(range(10))
                                ax_backprop.set_yticks(range(10))
                                # Minor ticks for grid lines
                                ax_backprop.set_xticks(np.arange(-0.5, 10.5, 1), minor=True)
                                ax_backprop.set_yticks(np.arange(-0.5, 10.5, 1), minor=True)
                                ax_backprop.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
                
                state = next_state
                valid_moves = next_valid_moves
                steps += 1
                total_steps += 1
                
                # Refresh UI
                fig.canvas.flush_events()
                time.sleep(state_vars['run_speed'])
                
                if done:
                    print(f"Episode {episode} finished. Steps={steps}. Winner={info.get('winner')}")
                    
                    # Curriculum Update
                    winner = info.get('winner')
                    pbs_acc = 0.8 # Mock accuracy since we don't have true labels easily in loop
                    if agent1.history and hasattr(agent1.history, 'avg_accuracy'):
                        pbs_acc = agent1.history.avg_accuracy
                        
                    curriculum.update_metrics({
                        'winner': winner,
                        'pbs_accuracy': pbs_acc,
                        'opponent_type': opponent_type
                    })
                    
                    fig.suptitle(f"Interactive Stratego Training (Phase {curriculum.current_phase.value}: {curriculum.current_phase.name})", fontsize=16)
            
            episode += 1
            
    except KeyboardInterrupt:
        print("Training stopped by user.")
    final_plt_show_block = getattr(plt, 'show', lambda: None)
    # plt.show() # Often blocking, careful


def visualize_board(ax, board, valid_moves, last_move=None, observed_board=None):
    """
    board: Ground Truth board (actual_board).
    observed_board: If provided, acts as a mask. 
                    - If a piece is in 'board' but 0 in 'observed_board', it is HIDDEN (Gray).
                    - If it's in both, it is REVEALED.
                    - If observed_board is None, all are REVEALED.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch
    
    board_np = board
    if isinstance(board, torch.Tensor):
        board_np = board.cpu().numpy()
        
    observed_np = None
    if observed_board is not None:
        observed_np = observed_board
        if isinstance(observed_board, torch.Tensor):
            observed_np = observed_board.cpu().numpy()
        
    ax.clear()
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    # Set minor ticks at mid-points for grid lines
    ax.set_xticks(np.arange(-0.5, 10.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10.5, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5) 
    ax.invert_yaxis() # Origin at top-left
    
    # Draw pieces
    for r in range(10):
        for c in range(10):
            val = board_np[r, c]
            
            if val == LAKE_SQUARE:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='cyan', alpha=0.3))
            elif val != 0:
                # Determine Identity and Visibility
                is_player = (val > 0)
                
                # Check Visibility
                is_revealed = True
                if observed_np is not None:
                    # If observed board has 0 at this location (and it's not a lake), it's hidden.
                    # But observed board might have 0 for lakes too. 
                    # Assuming board_np and observed_np are aligned.
                    obs_val = observed_np[r, c]
                    if obs_val == 0:
                        is_revealed = False
                
                # Color Logic
                color = 'gray'
                text_label = ""
                
                if is_player:
                    color = 'salmon'
                    try:
                        text_label = PIECE_NAMES.get(PieceType(abs(int(val))), str(abs(int(val))))
                    except ValueError:
                        text_label = str(abs(int(val)))
                else: # Enemy
                    if is_revealed:
                        color = 'lightsteelblue'
                        try:
                            text_label = PIECE_NAMES.get(PieceType(abs(int(val))), str(abs(int(val))))
                        except ValueError:
                            text_label = str(abs(int(val)))
                    else:
                        color = 'gray' # Hidden
                        text_label = "" # No text
                
                # Draw
                circle = plt.Circle((c, r), 0.4, color=color, alpha=0.9, zorder=2)
                ax.add_patch(circle)
                if text_label:
                    ax.text(c, r, text_label, color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # Highlight Last Move
    if last_move:
        (r1, c1), (r2, c2) = last_move
        # Draw Arrow
        # arrow = FancyArrowPatch((c1, r1), (c2, r2), arrowstyle='->', mutation_scale=20, color='gold', linewidth=2, zorder=4)
        # ax.add_patch(arrow)
        
        # Highlight Source and Dest
        ax.add_patch(plt.Rectangle((c1-0.5, r1-0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3, zorder=5))
        ax.add_patch(plt.Rectangle((c2-0.5, r2-0.5), 1, 1, fill=False, edgecolor='gold', linewidth=3, zorder=5))

def visualize_top_moves(ax, top_moves, search_info=None):
    """
    Render a text list of top moves.
    top_moves: list of ((r1, c1), (r2, c2), q_val) or similiar.
    # Actually VisualDQNAgent stores (move, q).
    move is ((r1, c1), (r2, c2)).
    
    search_info: Optional dict with search results to display.
    """
    ax.clear()
    ax.axis('off')
    
    # Title with search indicator
    title = "Top 10 Candidate Moves"
    if search_info and search_info.get('search_changed_decision'):
        title = " SEARCH REFINED"
    elif search_info:
        title = " Search Active"
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Header
    ax.text(0, 1.0, f"{'Move':<10} {'Q-Value':>10}", transform=ax.transAxes, fontsize=9, fontfamily='monospace', fontweight='bold')
    
    y = 0.9
    for i, (move, q) in enumerate(top_moves):
        (r1, c1), (r2, c2) = move
        move_str = f"({r1},{c1})->({r2},{c2})"
        q_str = f"{q:.3f}"
        
        # Highlight top move
        color = 'black'
        weight = 'normal'
        if i == 0:
            color = 'darkgreen'
            weight = 'bold'
            
        ax.text(0, y, f"{move_str:<12} {q_str:>8}", transform=ax.transAxes, fontsize=9, fontfamily='monospace', color=color, fontweight=weight)
        y -= 0.08
    
    # Search info box
    if search_info:
        y -= 0.05
        ax.axhline(y=y + 0.03, xmin=0, xmax=0.9, color='gray', linewidth=0.5, transform=ax.transAxes)
        ax.text(0, y, "─── Search Info ───", transform=ax.transAxes, fontsize=8, color='gray')
        y -= 0.06
        ax.text(0, y, f"Depth: {search_info.get('search_depth', 'N/A')}", transform=ax.transAxes, fontsize=8)
        y -= 0.05
        ax.text(0, y, f"Expanded: {search_info.get('moves_expanded', 'N/A')} moves", transform=ax.transAxes, fontsize=8)
        y -= 0.05
        changed = search_info.get('search_changed_decision', False)
        ax.text(0, y, f"Decision Changed: {'YES ' if changed else 'No'}", 
                transform=ax.transAxes, fontsize=8, color='blue' if changed else 'gray')


if __name__ == "__main__":
    visual_train()
