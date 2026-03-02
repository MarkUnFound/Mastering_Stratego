import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
import os
import sys
import time

# Add repository root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import StrategoEnvironment
from drqn_agent import RainbowAgent
from piece import PieceType, PIECE_NAMES, PIECE_RANKS
from board import BOARD_SIZE, LAKE_SQUARE

class DQNDashboard:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize environment and agent
        self.env = StrategoEnvironment(self.device)
        self.agent = RainbowAgent(player_id=1, device=self.device)
        
        if os.path.exists(model_path):
            print(f"Loading Model: {model_path}")
            self.agent.load_model(model_path)
        else:
            print(f"Warning: Model not found at {model_path}. Using random initialization.")

        # Ensure agent is in eval mode but hooks work
        self.agent.q_network.eval()
        
        # Hooks and Data Cache
        self.attn_weights = None
        self.hook_handle = None
        self._register_hooks()
        
        self.current_state = None
        self.selected_pos = (0, 0)  # Default selection (often a Marshall/General in setup)
        self.hover_pos = None
        self.q_heatmap = np.zeros((10, 10))
        self.attn_map = np.zeros((10, 10))
        self.selected_action_probs = None
        
        # PCA for AAREN embeddings
        self.pca = PCA(n_components=2)
        
        # Setup Figure
        plt.ion()
        plt.style.use('dark_background') # Global dark theme
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[1.2, 1])
        
        self.ax_board = self.fig.add_subplot(self.gs[0, 0])
        self.ax_c51 = self.fig.add_subplot(self.gs[0, 1])
        self.ax_q_heatmap = self.fig.add_subplot(self.gs[0, 2])
        self.ax_attn = self.fig.add_subplot(self.gs[1, 0])
        self.ax_aaren = self.fig.add_subplot(self.gs[1, 1:])
        
        for ax in [self.ax_board, self.ax_c51, self.ax_q_heatmap, self.ax_attn, self.ax_aaren]:
            ax.set_facecolor('#2c3e50')
        
        # Initialize Images and Colorbars once to avoid distortion
        self.im_q = self.ax_q_heatmap.imshow(np.zeros((10, 10)), cmap='magma', interpolation='nearest', vmin=0, vmax=5)
        self.cb_q = plt.colorbar(self.im_q, ax=self.ax_q_heatmap, fraction=0.046, pad=0.04)
        
        self.im_attn = self.ax_attn.imshow(np.zeros((10, 10)), cmap='viridis', interpolation='bilinear', vmin=0, vmax=0.2)
        self.cb_attn = plt.colorbar(self.im_attn, ax=self.ax_attn, fraction=0.046, pad=0.04)
        
        self.pca_scatter = self.ax_aaren.scatter([], [], c='#e67e22', s=50, alpha=0.7, edgecolors='white')
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self.running = True
        self.paused = True
        self.step_requested = False
        
        print("\nControls:")
        print("  Space: Pause/Resume")
        print("  Right Arrow: Step Single Turn")
        print("  Click Board: Select Piece for Attention/C51 View")
        print("  Close Window: Exit")

    def _register_hooks(self):
        """Register forward hook to extract attention weights."""
        def hook_fn(module, input, output):
            # output of nn.MultiheadAttention is (attn_output, attn_output_weights)
            _, weights = output
            self.attn_weights = weights.detach().cpu().numpy()
            
        target_layer = self.agent.q_network.spatial_attention.attn
        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        print("[OK] SpatialAttention hook registered.")

    def _on_click(self, event):
        if event.inaxes in [self.ax_board, self.ax_q_heatmap, self.ax_attn]:
            c, r = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= r < 10 and 0 <= c < 10:
                self.selected_pos = (r, c)
                self.update_plots()

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            print(f"Pausing: {self.paused}")
        elif event.key == 'right':
            self.step_requested = True
            self.paused = True

    def update_data(self):
        """Pulls fresh data from the agent/env."""
        with torch.no_grad():
            self.current_state = self.env.get_state()
            state_tensor = self.agent.get_state_representation(self.current_state, pbs_instance=self.agent.history)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Forward pass to trigger hooks and get logits
            log_probs = self.agent.q_network(state_tensor)
            probs = log_probs.exp()
            
            # 1. Expected Q-Values for Heatmap (Inferred per square)
            self.q_heatmap = np.full((10, 10), np.nan)
            valid_moves = self.env.get_valid_moves()
            expected_q_values = (probs * self.agent.support).sum(dim=2).squeeze(0).cpu().numpy()
            
            for move in valid_moves:
                (r1, c1), (r2, c2) = move
                action_idx = self.agent._move_to_action_index(move)
                q_val = expected_q_values[action_idx]
                if np.isnan(self.q_heatmap[r1, c1]) or q_val > self.q_heatmap[r1, c1]:
                    self.q_heatmap[r1, c1] = q_val
            
            # 2. C51 Atoms for selected action
            self.selected_action_probs = None
            best_q = -float('inf')
            for move in valid_moves:
                (r1, c1), (r2, c2) = move
                if (r1, c1) == self.selected_pos:
                    action_idx = self.agent._move_to_action_index(move)
                    q_val = expected_q_values[action_idx]
                    if q_val > best_q:
                        best_q = q_val
                        self.selected_action_probs = probs[0, action_idx].cpu().numpy()
            
            # 3. Attention Map
            if self.attn_weights is not None:
                sel_idx = self.selected_pos[0] * 10 + self.selected_pos[1]
                self.attn_map = self.attn_weights[0, sel_idx].reshape(10, 10)

            # 4. AAREN Embeddings
            self.embeddings = self.agent.history.get_embedding_tensor().cpu().numpy()
            emb_flat = self.embeddings.reshape(64, 100).T
            
            # Filter out squares with no history (all-zero embeddings)
            active_mask = np.any(np.abs(emb_flat) > 1e-5, axis=1)
            if np.sum(active_mask) >= 2:
                emb_active = emb_flat[active_mask]
                self.pca_coords = self.pca.fit_transform(emb_active)
                self.pca_indices = np.where(active_mask)[0]
            else:
                self.pca_coords = None

    def update_plots(self):
        self.update_data()
        
        # Clear specific parts (faster than full clear)
        self.ax_board.clear()
        self.ax_c51.clear()
        # Heatmap and Attn don't need full clear, we use set_data
        # But we clear text/patches
        for coll in self.ax_q_heatmap.collections: coll.remove() # remove any text if we added some
        for coll in self.ax_attn.collections: coll.remove()
        for art in self.ax_q_heatmap.texts: art.remove()
        for art in self.ax_attn.texts: art.remove()
        for patch in self.ax_q_heatmap.patches: patch.remove()
        for patch in self.ax_attn.patches: patch.remove()
        
        self.ax_aaren.clear()
        
        # 1. Main Board
        board = self.current_state.board
        actual_board = self.env.board.actual_board.cpu().numpy()
        
        # Draw background grid
        self.ax_board.set_facecolor('#2c3e50')
        self.ax_board.set_xlim(-0.5, 9.5)
        self.ax_board.set_ylim(9.5, -0.5)
        self.ax_board.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax_board.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax_board.grid(which='minor', color='#34495e', linestyle='-', linewidth=1)
        
        for r in range(10):
            for c in range(10):
                val = actual_board[r, c]
                if val == LAKE_SQUARE:
                    self.ax_board.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#2980b9', alpha=0.8))
                elif val != 0:
                    is_p1 = val > 0
                    color = '#2ecc71' if is_p1 else '#e74c3c'
                    
                    # Determine label and "Known" status
                    label = "?"
                    is_unknown = False
                    if not is_p1:
                        # Inferred by AAREN?
                        predictions = self.agent.history.get_piece_predictions((r, c))
                        if predictions:
                            most_likely_pt, conf = max(predictions.items(), key=lambda x: x[1])
                            pt_name = PIECE_NAMES.get(most_likely_pt, "???")[:3]
                            if conf > 0.4:
                                label = pt_name
                            else:
                                label = "?"
                            is_unknown = True
                        else:
                            label = "?"
                            is_unknown = True
                    else:
                        pt = PieceType(abs(int(val)))
                        label = PIECE_NAMES.get(pt, "?")[:3]
                    
                    # Draw piece
                    rect_color = color if not is_unknown else '#95a5a6'
                    self.ax_board.add_patch(plt.Rectangle((c-0.4, r-0.4), 0.8, 0.8, color=rect_color, alpha=0.9))
                    self.ax_board.text(c, r, label, ha='center', va='center', color='white', 
                                     fontsize=7, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

        self.ax_board.set_title("Stratego Board (Agent Perception)", color='white', weight='bold', fontsize=12)
        self.ax_board.add_patch(plt.Rectangle((self.selected_pos[1]-0.5, self.selected_pos[0]-0.5), 1, 1, fill=False, edgecolor='#f1c40f', lw=3))

        # Add piece rank legend (numbers context)
        legend_text = (
            "Piece Rank Legend:\n"
            "F: Flag (1)\n"
            "1: Spy (2)\n"
            "2: Scout (3)\n"
            "3: Miner (4)\n"
            "4-8: Sgt-Col\n"
            "9: General (10)\n"
            "M: Marshal (11)\n"
            "B: Bomb (12)"
        )
        self.ax_board.text(10.5, 5, legend_text, color='white', fontsize=9, 
                          va='center', ha='left', bbox=dict(facecolor='#34495e', alpha=0.8, pad=5))

        # 2. C51 PDF
        if self.selected_action_probs is not None:
            atoms = self.agent.support.cpu().numpy()
            self.ax_c51.fill_between(atoms, self.selected_action_probs, color='#9b59b6', alpha=0.7)
            self.ax_c51.plot(atoms, self.selected_action_probs, color='#8e44ad', lw=2)
            self.ax_c51.set_xlabel("Return (Q)", color='white')
            self.ax_c51.set_ylim(0, max(0.2, np.max(self.selected_action_probs)*1.2))
            self.ax_c51.grid(True, alpha=0.3)
        else:
            self.ax_c51.text(0.5, 0.5, "No Valid Moves\nor Square Empty", ha='center', va='center', 
                             transform=self.ax_c51.transAxes, color='gray')
            
        self.ax_c51.set_title(f"Outcome Prob (C51) for {self.selected_pos}", color='white', fontsize=10)

        # 3. Q-Heatmap
        masked_q = np.ma.masked_where(np.isnan(self.q_heatmap), self.q_heatmap)
        self.im_q.set_data(masked_q)
        # Auto-scale q-heatmap if values exist
        if not np.all(np.isnan(self.q_heatmap)):
            self.im_q.set_clim(np.nanmin(self.q_heatmap), np.nanmax(self.q_heatmap))
        self.ax_q_heatmap.set_title("Strategic Piece Priority (Max Q-Source)", color='white', fontsize=10)

        # 4. Spatial Attention
        self.im_attn.set_data(self.attn_map)
        if self.attn_map is not None:
            self.im_attn.set_clim(0, max(0.01, np.max(self.attn_map)))
        self.ax_attn.set_title(f"Focus Attention from {self.selected_pos}", color='white', fontsize=10)
        self.ax_attn.add_patch(plt.Circle((self.selected_pos[1], self.selected_pos[0]), 0.3, color='white', fill=False, lw=2))

        # 5. AAREN PCA (Enemy Identity)
        self.ax_aaren.set_facecolor('#2c3e50')
        if self.pca_coords is not None:
            # Re-draw scatter as it was cleared
            self.ax_aaren.scatter(self.pca_coords[:, 0], self.pca_coords[:, 1], c='#e67e22', s=60, alpha=0.9, edgecolors='white', zorder=3)
            for i in range(len(self.pca_coords)):
                idx = self.pca_indices[i]
                r, c = idx // 10, idx % 10
                self.ax_aaren.annotate(f"({r},{c})", (self.pca_coords[i, 0], self.pca_coords[i, 1]), 
                                     fontsize=9, xytext=(5, 5), textcoords='offset points', 
                                     color='#f1c40f', weight='bold', zorder=4)
            self.ax_aaren.grid(True, alpha=0.3, zorder=1)
        else:
            self.ax_aaren.text(0.5, 0.5, "Insufficient Action History\n(Wait for enemy moves)", 
                             ha='center', va='center', transform=self.ax_aaren.transAxes, color='#95a5a6', fontsize=12)
        
        self.ax_aaren.set_title("Enemy Identity Inference (Behavioral Clusters)", color='white', fontsize=12, weight='bold')

        # Formatting Fixes
        self.fig.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def step(self):
        """Advances the game by one turn."""
        current_player = self.env.current_player
        valid_moves = self.env.get_valid_moves()
        
        if not valid_moves:
            print("GAME OVER - No valid moves")
            return True
        
        # Choose action
        if current_player == 1:
            # DQN Agent
            action = self.agent.act(self.env.get_state(), valid_moves)
        else:
            # Simple heuristic/random opponent for demo
            action = valid_moves[np.random.randint(len(valid_moves))]
            
        if action:
            next_state, reward, done, info = self.env.step(action)
            
            # Update AAREN for BOTH players (since AAREN is part of the agent's observation)
            # Both players observe the action and update their internal history
            # But the agent only tracks ENEMY actions.
            
            # If P2 moved, P1 (the agent) updates its history
            if current_player == -1:
                self.agent.history.update(action, self.env.get_state(), acting_player=-1)
                
            # If battle occurred, update from reveal
            revealed = info.get('revealed_in_step', [])
            for pos, piece_type in revealed:
                # Agent (P1) only tracks enemy (P2 < 0) reveals for training/inference
                # But history aggregator update handles filtering if needed.
                self.agent.history.update_from_reveal(pos, piece_type)
            
            return done
        return True

    def run(self):
        """Main loop."""
        print("Starting Dashboard Loop...")
        self.env.reset()
        self.agent.reset_history()
        self.update_plots()
        
        last_time = time.time()
        
        while self.running:
            if not self.paused or self.step_requested:
                if time.time() - last_time > 0.5 or self.step_requested:
                    done = self.step()
                    self.update_plots()
                    last_time = time.time()
                    self.step_requested = False
                    
                    if done:
                        print("Resetting game...")
                        time.sleep(1)
                        self.env.reset()
                        self.agent.reset_history()
            
            plt.pause(0.01)
            
            if not plt.fignum_exists(self.fig.number):
                self.running = False
                
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    # Path to model
    model_path = r"c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\agent1_rainbow_episode_1000.pt"
    
    dashboard = DQNDashboard(model_path)
    dashboard.run()
