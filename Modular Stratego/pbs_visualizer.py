# stratego_modular/pbs_visualizer.py

"""
PBS Visualization Module - Production Hybrid Version
Handles visualization of Probabilistic Belief State (PBS) for Stratego agents.

FEATURES:
1. Top-3 predictions with visual hierarchy (top prediction emphasized)
2. Uncertainty-based border width using Shannon entropy (1.0-5.0pt range)
3. Color-coded probabilities for better readability
4. Intelligent filtering to reduce visual clutter
"""

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import os
from typing import Optional, Tuple, Dict, List
from piece import PieceType, PIECE_RANKS, PIECE_NAMES
from board import BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE, HIDDEN_PIECE


def _calculate_entropy(beliefs: dict) -> float:
    """
    Calculate Shannon entropy of a belief distribution.
    
    Uses information-theoretic standard (log base 2) to measure uncertainty.
    Higher entropy = more uncertainty (uniform distribution)
    Lower entropy = more certainty (peaked distribution)
    
    Args:
        beliefs: Dictionary mapping PieceType to probability
        
    Returns:
        Entropy value in bits (0 to log2(num_types))
    """
    if not beliefs:
        return 0.0
    
    entropy = 0.0
    for prob in beliefs.values():
        if prob > 1e-10:  # Avoid log(0)
            entropy -= prob * np.log2(prob)
    
    return entropy


def _get_border_width_from_entropy(entropy: float, max_entropy: float) -> float:
    """
    Convert entropy to border width for visual uncertainty indication.
    
    Maps normalized entropy to a border width range that provides
    strong visual contrast between confident and uncertain predictions.
    
    Low entropy (certain) -> thin border (1.0pt)
    High entropy (uncertain) -> thick border (5.0pt)
    
    Args:
        entropy: Current entropy value
        max_entropy: Maximum possible entropy (log2(num_piece_types))
        
    Returns:
        Border width in range [1.0, 5.0] points
    """
    if max_entropy < 1e-10:
        return 1.5  # Default fallback
    
    # Normalize entropy to [0, 1]
    normalized = entropy / max_entropy
    
    # Map to border width: [1.0, 5.0]
    # 4.0pt range provides strong visual distinction
    min_width = 1.0
    max_width = 5.0
    width = min_width + normalized * (max_width - min_width)
    
    return width


def _get_top3_predictions(pbs, pos: tuple) -> List[Tuple[PieceType, float]]:
    """
    Extract top-3 PBS predictions for a position with their probabilities.
    
    Returns predictions sorted by probability (highest first).
    Useful for showing alternative hypotheses beyond the single best prediction.
    
    ENHANCED: Now handles empty beliefs by creating uniform distribution fallback
    to ensure visualization always shows predictions instead of '?'.
    
    Args:
        pbs: ProbabilisticBeliefState object
        pos: Position tuple (r, c)
        
    Returns:
        List of tuples: [(piece_type, probability), ...] up to 3 items
    """
    # Ensure beliefs are initialized for this position (handles both defaultdict and regular dict)
    if pos not in pbs.belief_distributions:
        if hasattr(pbs.belief_distributions, '__getitem__'):
            # This is a defaultdict or dict - access to create if defaultdict
            try:
                beliefs = pbs.belief_distributions[pos]
            except (KeyError, TypeError):
                # Regular dict - initialize manually
                pbs.belief_distributions[pos] = {}
                beliefs = pbs.belief_distributions[pos]
        else:
            # Unexpected type - initialize as empty dict
            beliefs = {}
    else:
        beliefs = pbs.belief_distributions[pos]
    
    if not beliefs:
        # If beliefs dictionary is empty, create uniform distribution as fallback
        # This ensures visualization always shows predictions instead of '?'
        uniform_prob = 1.0 / len(PieceType)
        predictions = [(piece_type, uniform_prob) for piece_type in PieceType]
        # Sort by piece value and return top-3
        sorted_predictions = sorted(predictions, key=lambda x: x[0].value)
        return sorted_predictions[:3]
    
    # Sort by probability (descending), then by piece value for ties
    sorted_beliefs = sorted(beliefs.items(), key=lambda x: (-x[1], x[0].value))
    
    # Return top-3
    return [(piece_type, prob) for piece_type, prob in sorted_beliefs[:3]]


def visualize_pbs_state_as_image(
    actual_board,
    agent1_pbs,
    agent2_pbs,
    episode: int,
    move_num: int = 0,
    visible_board_p1: Optional = None,
    visible_board_p2: Optional = None
):
    """
    Visualize PBS state and return as PIL Image (for GIF creation).
    
    Similar to visualize_pbs_state but returns PIL Image instead of saving to file.
    
    Args:
        actual_board: Actual board state
        agent1_pbs: Agent 1's PBS object
        agent2_pbs: Agent 2's PBS object
        episode: Episode number
        move_num: Move number for title
        visible_board_p1: Optional visible board for player 1
        visible_board_p2: Optional visible board for player 2
        
    Returns:
        PIL Image of the PBS visualization
    """
    import io
    from PIL import Image
    
    # Create the visualization
    fig = _create_pbs_figure(
        actual_board,
        agent1_pbs,
        agent2_pbs,
        episode,
        move_num,
        visible_board_p1,
        visible_board_p2
    )
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')
    plt.close(fig)
    
    return img


def create_pbs_gif(pbs_states: List[Dict], episode: int, save_path: str, frame_duration: int = 750):
    """
    Create a GIF from a sequence of PBS states recorded during an episode.
    
    Args:
        pbs_states: List of dicts with keys: 'actual_board', 'agent1_pbs', 'agent2_pbs', 
                   'move_num', 'visible_board_p1', 'visible_board_p2'
        episode: Episode number
        save_path: Path to save the GIF
        frame_duration: Duration of each frame in milliseconds (default 750ms)
    """
    import io
    from PIL import Image
    
    try:
        if not pbs_states:
            print(f"⚠️  No PBS states to create GIF for episode {episode}")
            return
        
        frames = []
        print(f"🎬 Creating {len(pbs_states)} frames for PBS GIF...")
        
        for i, state in enumerate(pbs_states):
            try:
                actual_board = state['actual_board']
                agent1_pbs = state.get('agent1_pbs', None)
                agent2_pbs = state.get('agent2_pbs', None)
                move_num = state.get('move_num', i)
                visible_board_p1 = state.get('visible_board_p1', None)
                visible_board_p2 = state.get('visible_board_p2', None)
                
                # Create PBS visualization for this state
                img = visualize_pbs_state_as_image(
                    actual_board,
                    agent1_pbs,
                    agent2_pbs,
                    episode,
                    move_num,
                    visible_board_p1,
                    visible_board_p2
                )
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                frames.append(img)
                print(f"  Frame {i+1}/{len(pbs_states)} created (move {move_num})")
                
            except Exception as e:
                print(f"⚠️  Error creating frame {i}: {e}")
                continue
        
        if not frames:
            print(f"⚠️  No frames created for PBS GIF episode {episode}")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        print(f"💾 Saving GIF with {len(frames)} frames at {frame_duration}ms per frame...")
        
        # Save GIF with improved compatibility
        try:
            # First try with optimization
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,
                optimize=True,
                disposal=2  # Clear frame before next one
            )
        except Exception as e:
            print(f"⚠️  Optimization failed, trying without optimization: {e}")
            # Fallback without optimization
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,
                disposal=2  # Clear frame before next one
            )
        
        print(f"✅ PBS GIF created for episode {episode}: {save_path}")
        print(f"   Frame duration: {frame_duration}ms")
        print(f"   Total frames: {len(frames)}")
        
        # Verify the file was created and get its size
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"   File size: {file_size / 1024:.1f} KB")
        
    except ImportError as e:
        print(f"❌ PIL/Pillow not available: {e}")
        print("   Install with: pip install Pillow")
    except Exception as e:
        print(f"⚠️  Error creating PBS GIF for episode {episode}: {e}")
        import traceback
        traceback.print_exc()


def visualize_pbs_state(
    actual_board,
    agent1_pbs,
    agent2_pbs,
    episode: int,
    save_path: str,
    visible_board_p1: Optional = None,
    visible_board_p2: Optional = None
):
    """
    Visualize the Probabilistic Belief State (PBS) for both agents.
    
    Creates a 3-panel visualization showing:
    - Actual board with all pieces
    - Agent 1's PBS beliefs (what Agent 1 thinks about Agent 2's pieces)
    - Agent 2's PBS beliefs (what Agent 2 thinks about Agent 1's pieces)
    
    VISUAL FEATURES:
    - Top prediction: 2-line format (name + probability) for emphasis
    - 2nd/3rd predictions: Compact format, only shown if prob > 5%
    - Border thickness: 1.0pt (confident) to 5.0pt (uncertain)
    - Color coding: DarkBlue for probabilities, Gray for alternatives
    
    Args:
        actual_board: The actual game board (10x10 tensor)
        agent1_pbs: Agent 1's PBS object (or None if not available)
        agent2_pbs: Agent 2's PBS object (or None if not available)
        episode: Current episode number
        save_path: Path to save the visualization
        visible_board_p1: Visible board for player 1 (shows what player 1 can see)
        visible_board_p2: Visible board for player 2 (shows what player 2 can see)
    """
    fig = _create_pbs_figure(actual_board, agent1_pbs, agent2_pbs, episode, 
                            visible_board_p1=visible_board_p1, visible_board_p2=visible_board_p2)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"🎯 PBS visualization saved to {save_path}")


def _create_pbs_figure(
    actual_board,
    agent1_pbs,
    agent2_pbs,
    episode: int,
    move_num: int = 0,
    visible_board_p1: Optional = None,
    visible_board_p2: Optional = None
):
    """
    Create PBS visualization figure (without saving).
    
    Args:
        actual_board: The actual game board (10x10 tensor)
        agent1_pbs: Agent 1's PBS object (or None if not available)
        agent2_pbs: Agent 2's PBS object (or None if not available)
        episode: Current episode number
        move_num: Move number (0 if not specified)
        visible_board_p1: Visible board for player 1
        visible_board_p2: Visible board for player 2
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(20, 12))
    title = f'PBS Visualization - Episode {episode}'
    if move_num > 0:
        title += f' - Move {move_num}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Convert boards to numpy
    actual_board_np = actual_board.cpu().numpy() if hasattr(actual_board, 'cpu') else actual_board
    visible_board_p1_np = visible_board_p1.cpu().numpy() if visible_board_p1 is not None and hasattr(visible_board_p1, 'cpu') else visible_board_p1
    visible_board_p2_np = visible_board_p2.cpu().numpy() if visible_board_p2 is not None and hasattr(visible_board_p2, 'cpu') else visible_board_p2
    
    # Create subplots: Actual board, Agent 1 PBS, Agent 2 PBS
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    # Plot 1: Actual Board
    _plot_actual_board(ax1, actual_board_np, "Actual Board State")
    
    # Plot 2: Agent 1 PBS (use visible board for player 1 if available, otherwise use actual)
    visible_for_p1 = visible_board_p1_np if visible_board_p1_np is not None else actual_board_np
    if agent1_pbs:
        _plot_pbs_beliefs(ax2, visible_for_p1, actual_board_np, agent1_pbs, player_id=1, title="Agent 1 PBS Beliefs")
    else:
        ax2.text(0.5, 0.5, 'PBS Not Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Agent 1 PBS Beliefs")
    
    # Plot 3: Agent 2 PBS (use visible board for player 2 if available, otherwise use actual)
    visible_for_p2 = visible_board_p2_np if visible_board_p2_np is not None else actual_board_np
    if agent2_pbs:
        _plot_pbs_beliefs(ax3, visible_for_p2, actual_board_np, agent2_pbs, player_id=-1, title="Agent 2 PBS Beliefs")
    else:
        ax3.text(0.5, 0.5, 'PBS Not Available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title("Agent 2 PBS Beliefs")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def _plot_actual_board(ax, board_np: np.ndarray, title: str):
    """
    Plot the actual board with all pieces visible.
    
    Args:
        ax: Matplotlib axes object
        board_np: Board state as numpy array
        title: Title for the subplot
    """
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Draw board squares with cell borders
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece_val = board_np[r, c]
            
            # Determine square color
            if piece_val == LAKE_SQUARE:
                color = 'lightblue'
            elif piece_val == EMPTY_SQUARE:
                color = 'white'
            elif piece_val > 0:
                color = 'lightcoral'  # Player 1 (Red)
            else:
                color = 'lightgreen'  # Player 2 (Green)
            
            # Draw square with standard border
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                   facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            # Draw piece label
            if piece_val != EMPTY_SQUARE and piece_val != LAKE_SQUARE:
                piece_type = PieceType(abs(int(piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                text_color = 'black' if abs(piece_val) <= 6 else 'white'
                ax.text(c, r, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color=text_color)


def _plot_pbs_beliefs(ax, visible_board_np: np.ndarray, actual_board_np: np.ndarray, 
                     pbs, player_id: int, title: str):
    """
    Plot PBS beliefs for a specific agent with hybrid visual enhancements.
    
    HYBRID FEATURES:
    - Modular entropy calculation (improved code structure)
    - Wide border range 1.0-5.0pt (stronger visual contrast)
    - 2-line top prediction format (better hierarchy)
    - 5% threshold for alternatives (reduced clutter)
    - Color-coded display (improved readability)
    
    Args:
        ax: Matplotlib axes object
        visible_board_np: What this agent can see
        actual_board_np: Actual board state (for color coding)
        pbs: ProbabilisticBeliefState object
        player_id: 1 for Agent 1, -1 for Agent 2
        title: Title for the subplot
    """
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Safety check: ensure PBS object exists and has required attributes
    if pbs is None:
        ax.text(0.5, 0.5, 'PBS Not Available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return
    
    if not hasattr(pbs, 'belief_distributions'):
        ax.text(0.5, 0.5, 'PBS Not Initialized', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return
    
    # Calculate maximum possible entropy for normalization (global constant)
    max_entropy = np.log2(len(PieceType))  # ~3.585 bits for 12 piece types
    
    # Find all enemy piece positions from actual board
    enemy_positions = set()
    if actual_board_np is not None:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                try:
                    actual_val = actual_board_np[r, c]
                    if hasattr(actual_val, 'item'):
                        actual_piece_val = int(actual_val.item())
                    elif isinstance(actual_val, (np.integer, np.floating)):
                        actual_piece_val = int(actual_val)
                    else:
                        actual_piece_val = int(actual_val)
                    
                    # Check if this is an enemy piece
                    if actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                        if player_id == 1:
                            # Agent 1: enemy pieces are negative
                            if actual_piece_val < 0:
                                enemy_positions.add((r, c))
                        elif player_id == -1:
                            # Agent 2: enemy pieces are positive
                            if actual_piece_val > 0:
                                enemy_positions.add((r, c))
                except (ValueError, TypeError):
                    pass
    
    # Draw board squares with PBS information
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            pos = (r, c)
            
            # Extract values properly - handle both numpy arrays and tensors
            try:
                visible_val = visible_board_np[r, c]
                if hasattr(visible_val, 'item'):
                    visible_piece_val = int(visible_val.item())
                elif isinstance(visible_val, (np.integer, np.floating)):
                    visible_piece_val = int(visible_val)
                else:
                    visible_piece_val = int(visible_val)
            except (ValueError, TypeError):
                visible_piece_val = visible_board_np[r, c]
            
            # Extract actual board value - for determining piece ownership
            if actual_board_np is not None:
                try:
                    actual_val = actual_board_np[r, c]
                    if hasattr(actual_val, 'item'):
                        actual_piece_val = int(actual_val.item())
                    elif isinstance(actual_val, (np.integer, np.floating)):
                        actual_piece_val = int(actual_val)
                    else:
                        actual_piece_val = int(actual_val)
                except (ValueError, TypeError):
                    actual_piece_val = actual_board_np[r, c]
            else:
                actual_piece_val = visible_piece_val
            
            # Handle different square types
            if visible_piece_val == LAKE_SQUARE:
                # Lake squares
                color = 'lightblue'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                continue
            
            # Check if this is an own piece (using actual board to avoid HIDDEN_PIECE conflicts)
            is_own_piece_by_actual = False
            if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                if player_id == 1:
                    is_own_piece_by_actual = (actual_piece_val > 0)
                elif player_id == -1:
                    is_own_piece_by_actual = (actual_piece_val < 0)
            
            if is_own_piece_by_actual:
                # Own pieces - show actual piece
                color = 'lightcoral' if player_id == 1 else 'lightgreen'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                piece_type = PieceType(abs(int(actual_piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                ax.text(c, r, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black')
                continue
                
            elif visible_piece_val == EMPTY_SQUARE and pos not in enemy_positions:
                # Empty squares
                color = 'white'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                continue
                
            elif visible_piece_val == HIDDEN_PIECE or pos in enemy_positions:
                # Hidden enemy pieces - show PBS beliefs
                
                # Verify this is actually an enemy piece
                is_enemy_piece = (pos in enemy_positions)
                
                if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                    if player_id == 1:
                        is_actually_enemy = (actual_piece_val < 0)
                    elif player_id == -1:
                        is_actually_enemy = (actual_piece_val > 0)
                    else:
                        is_actually_enemy = False
                else:
                    is_actually_enemy = False
                
                # Only show PBS beliefs if this is actually an enemy piece
                if not is_enemy_piece or not is_actually_enemy or actual_piece_val == EMPTY_SQUARE:
                    color = 'white'
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                           facecolor=color, edgecolor='black', linewidth=1.5)
                    ax.add_patch(rect)
                    continue
                
                # CRITICAL FIX: Ensure beliefs are initialized for ALL enemy positions
                # This ensures scouts and all enemy pieces are displayed, even if no actions observed yet
                if pos not in pbs.belief_distributions:
                    # Trigger defaultdict to create beliefs if they don't exist
                    _ = pbs.belief_distributions[pos]
                
                beliefs = pbs.belief_distributions[pos]
                
                # If beliefs dictionary is empty, initialize with uniform distribution as fallback
                if not beliefs:
                    uniform_prob = 1.0 / len(PieceType)
                    for piece_type in PieceType:
                        pbs.belief_distributions[pos][piece_type] = uniform_prob
                    beliefs = pbs.belief_distributions[pos]
                
                # Get PBS predictions and calculate entropy
                # CRITICAL FIX: Check if piece is actually visible to this agent, not just in revealed_pieces
                # revealed_pieces tracks pieces revealed during battles, but we need to check visibility
                revealed_pieces = getattr(pbs, 'revealed_pieces', {})
                is_piece_revealed = (pos in revealed_pieces)
                
                # Also check if piece is visible on the visible board (not HIDDEN_PIECE)
                # CRITICAL: At setup time, enemy pieces should be HIDDEN_PIECE, so is_piece_visible should be False
                is_piece_visible = (visible_piece_val != HIDDEN_PIECE and 
                                   visible_piece_val != EMPTY_SQUARE and 
                                   visible_piece_val != LAKE_SQUARE)
                
                # Only treat as revealed if it's actually visible to this agent AND in revealed_pieces
                # At setup time, pieces are HIDDEN_PIECE even though they exist, so we show predictions
                # ADDITIONAL CHECK: Ensure revealed_pieces is not empty (setup time check)
                # If revealed_pieces is empty, we're at setup and should show predictions, not revealed piece
                is_at_setup = (len(revealed_pieces) == 0)
                
                if is_piece_revealed and is_piece_visible and not is_at_setup:
                    # Revealed piece - use the stored revealed piece type from PBS (most reliable)
                    # CRITICAL FIX: Use revealed_pieces[pos] which stores the correct PieceType from update_from_reveal
                    revealed_piece_type = revealed_pieces[pos]
                    
                    # VERIFICATION: Double-check that revealed_piece_type matches actual board value
                    # This ensures PBS is using the correct information
                    if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                        actual_piece_type = PieceType(abs(int(actual_piece_val)))
                        # If there's a mismatch, use actual board value (ground truth) instead
                        if revealed_piece_type != actual_piece_type:
                            # PBS has wrong information - use actual board value
                            revealed_piece_type = actual_piece_type
                    
                    # Set up for revealed piece display
                    top3_predictions = []  # No predictions needed - we know the piece type
                    border_width = 1.5
                    confidence_normalized = 1.0
                    confidence_rgb = np.array([0.8, 0.8, 0.8], dtype=np.float64)
                    
                    # Determine color based on actual piece ownership
                    if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                        if actual_piece_val > 0:
                            color = 'lightcoral'
                        elif actual_piece_val < 0:
                            color = 'lightgreen'
                        else:
                            color = 'lightcoral' if visible_piece_val > 0 else 'lightgreen'
                    else:
                        color = 'lightcoral' if visible_piece_val > 0 else 'lightgreen'
                    
                    # Draw rectangle
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                           facecolor=color, edgecolor='black', linewidth=border_width)
                    ax.add_patch(rect)
                    
                    # Display the revealed piece type (verified against actual board)
                    piece_name = PIECE_NAMES.get(revealed_piece_type, '?')
                    ax.text(c, r - 0.15, piece_name, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')
                    ax.text(c, r + 0.15, f'C:1.00', ha='center', va='center', 
                           fontsize=8, color='darkblue', fontweight='bold')
                    
                    # Skip the rest of the prediction display logic
                    continue
                else:
                    # Hidden piece - show PBS predictions (even if in revealed_pieces but not visible)
                    # This handles cases where revealed_pieces might be incorrectly populated
                    top3_predictions = _get_top3_predictions(pbs, pos)
                    
                    # Calculate entropy and border width (beliefs already initialized above)
                    entropy = _calculate_entropy(beliefs)
                    border_width = _get_border_width_from_entropy(entropy, max_entropy)
                    
                    # Get highest confidence for color coding
                    highest_confidence = top3_predictions[0][1] if top3_predictions else 0.0
                    # Use actual probability without artificial clamping to maintain display logic
                    confidence_normalized = highest_confidence
                    
                    # Safety check: verify enemy ownership
                    if player_id == -1 and actual_piece_val <= 0:
                        color = 'white'
                        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                               facecolor=color, edgecolor='black', linewidth=border_width)
                        ax.add_patch(rect)
                        continue
                    elif player_id == 1 and actual_piece_val >= 0:
                        color = 'white'
                        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                               facecolor=color, edgecolor='black', linewidth=border_width)
                        ax.add_patch(rect)
                        continue
                    
                    # Color intensity based on confidence
                    rgba = plt.cm.RdYlGn(confidence_normalized)
                    confidence_rgb = np.array(rgba[:3], dtype=np.float64)
                
                # Determine base color based on actual piece ownership
                if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                    if actual_piece_val > 0:
                        # Agent 1's piece - red
                        base_rgb = np.array(mcolors.to_rgb('lightcoral'), dtype=np.float64)
                        blended_color = 0.85 * base_rgb + 0.15 * confidence_rgb
                        color = tuple(blended_color)
                    elif actual_piece_val < 0:
                        # Agent 2's piece - green
                        base_rgb = np.array(mcolors.to_rgb('lightgreen'), dtype=np.float64)
                        blended_color = 0.85 * base_rgb + 0.15 * confidence_rgb
                        color = tuple(blended_color)
                    else:
                        color = tuple(confidence_rgb)
                else:
                    color = tuple(confidence_rgb)
                
                # HYBRID ENHANCEMENT: Variable border width based on entropy
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=border_width)
                ax.add_patch(rect)
                
                # HYBRID ENHANCEMENT: Top-3 predictions with visual hierarchy
                # Always show predictions if available (even with low confidence)
                # This ensures uniform distributions are displayed instead of '?'
                if top3_predictions and len(top3_predictions) > 0:
                    # Display top prediction (2-line format for emphasis)
                    top_piece_type, top_prob = top3_predictions[0]
                    top_piece_name = PIECE_NAMES.get(top_piece_type, '?')
                    
                    # Display predictions with proper hierarchy - always show all available predictions
                    # This ensures Episode 1 style display continues throughout training
                    top_piece_type, top_prob = top3_predictions[0]
                    top_piece_name = PIECE_NAMES.get(top_piece_type, '?')
                    
                    # Always show the full hierarchical display for consistency
                    ax.text(c, r - 0.25, top_piece_name, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')
                    ax.text(c, r - 0.05, f'{top_prob:.2f}', ha='center', va='center', 
                           fontsize=7, color='darkblue', fontweight='bold')
                    
                    # Display 2nd and 3rd predictions (compact format, only if they exist and have reasonable probability)
                    # CRITICAL FIX: Always show top-3 predictions if available, regardless of probability threshold
                    # This ensures we see the full distribution, especially at setup time
                    if len(top3_predictions) > 1:
                        second_type, second_prob = top3_predictions[1]
                        second_name = PIECE_NAMES.get(second_type, '?')[:3]  # Abbreviate
                        ax.text(c, r + 0.15, f"{second_name}:{second_prob:.2f}", 
                               ha='center', va='center', fontsize=6, color='gray')
                    
                    if len(top3_predictions) > 2:
                        third_type, third_prob = top3_predictions[2]
                        third_name = PIECE_NAMES.get(third_type, '?')[:3]  # Abbreviate
                        ax.text(c, r + 0.28, f"{third_name}:{third_prob:.2f}", 
                               ha='center', va='center', fontsize=6, color='gray')
                else:
                    # Emergency fallback: ensure predictions are always shown instead of '?'
                    # Get predictions with guaranteed non-empty result
                    emergency_predictions = _get_top3_predictions(pbs, pos)
                    
                    if emergency_predictions:
                        # Display emergency predictions
                        top_piece_type, top_prob = emergency_predictions[0]
                        top_piece_name = PIECE_NAMES.get(top_piece_type, '?')
                        
                        ax.text(c, r - 0.25, top_piece_name, ha='center', va='center', 
                               fontsize=10, fontweight='bold', color='black')
                        ax.text(c, r - 0.05, f'{top_prob:.2f}', ha='center', va='center', 
                               fontsize=7, color='darkblue', fontweight='bold')
                        
                        # Show alternatives if available (always show if they exist, even with low probability)
                        if len(emergency_predictions) > 1:
                            second_type, second_prob = emergency_predictions[1]
                            second_name = PIECE_NAMES.get(second_type, '?')[:3]
                            ax.text(c, r + 0.15, f"{second_name}:{second_prob:.2f}", 
                                   ha='center', va='center', fontsize=6, color='gray')
                        
                        if len(emergency_predictions) > 2:
                            third_type, third_prob = emergency_predictions[2]
                            third_name = PIECE_NAMES.get(third_type, '?')[:3]
                            ax.text(c, r + 0.28, f"{third_name}:{third_prob:.2f}", 
                                   ha='center', va='center', fontsize=6, color='gray')
                    else:
                        # Absolute last resort: show uniform predictions
                        uniform_prob = 1.0 / len(PieceType)
                        ax.text(c, r - 0.25, 'F', ha='center', va='center', 
                               fontsize=10, fontweight='bold', color='black')
                        ax.text(c, r - 0.05, f'{uniform_prob:.2f}', ha='center', va='center', 
                               fontsize=7, color='darkblue', fontweight='bold')
            else:
                # Revealed enemy piece - use actual revealed piece type from PBS or actual board
                # CRITICAL FIX: Use actual_piece_val (ground truth) instead of visible_piece_val
                # visible_piece_val might be from wrong perspective or incorrect
                if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                    if actual_piece_val > 0:
                        color = 'lightcoral'
                    elif actual_piece_val < 0:
                        color = 'lightgreen'
                    else:
                        color = 'lightcoral' if visible_piece_val > 0 else 'lightgreen'
                    
                    # Use actual_piece_val to get the correct piece type (ground truth)
                    piece_type = PieceType(abs(int(actual_piece_val)))
                else:
                    # Fallback: try to get from revealed_pieces dictionary
                    revealed_pieces = getattr(pbs, 'revealed_pieces', {})
                    if pos in revealed_pieces:
                        piece_type = revealed_pieces[pos]
                    else:
                        # Last resort: use visible_piece_val (may be incorrect)
                        piece_type = PieceType(abs(int(visible_piece_val)))
                    
                    color = 'lightcoral' if visible_piece_val > 0 else 'lightgreen'
                
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                
                piece_name = PIECE_NAMES.get(piece_type, '?')
                ax.text(c, r - 0.15, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black')
                ax.text(c, r + 0.15, f'C:1.00', ha='center', va='center', 
                       fontsize=8, color='darkblue', fontweight='bold')


def _get_pbs_prediction(pbs, pos: tuple) -> tuple:
    """
    Get PBS prediction for a position based ONLY on beliefs.
    
    This is a legacy helper function maintained for backward compatibility.
    For production use, prefer _get_top3_predictions() for richer information.
    
    Args:
        pbs: ProbabilisticBeliefState object
        pos: Position tuple (r, c)
        
    Returns:
        Tuple of (piece_type, confidence)
    """
    uniform_prob = 1.0 / len(PieceType)
    
    if pos in pbs.belief_distributions:
        beliefs = pbs.belief_distributions[pos]
        belief_values = list(beliefs.values())
        
        if len(belief_values) > 0:
            min_val = min(belief_values)
            max_val = max(belief_values)
            is_uniform = (max_val - min_val) < 1e-6
            
            if is_uniform:
                return PieceType.FLAG, uniform_prob
            else:
                sorted_beliefs = sorted(beliefs.items(), key=lambda x: (-x[1], x[0].value))
                piece_type, confidence = sorted_beliefs[0]
                
                if confidence <= uniform_prob * 1.15:
                    return PieceType.FLAG, uniform_prob
                elif piece_type == PieceType.SPY and confidence < uniform_prob * 1.5:
                    return PieceType.FLAG, uniform_prob
                else:
                    return piece_type, confidence
        else:
            return PieceType.FLAG, uniform_prob
    else:
        return PieceType.FLAG, uniform_prob