# stratego_modular/pbs_visualizer.py

"""
PBS Visualization Module
Handles visualization of Probabilistic Belief State (PBS) for Stratego agents.
"""

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import os
from typing import Optional, Tuple
from .piece import PieceType, PIECE_NAMES, PIECE_RANKS
from .board import BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE, HIDDEN_PIECE


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
    
    Shows:
    - Actual board with pieces
    - Agent 1's PBS beliefs (inferred values and confidence) - using visible board for Agent 1
    - Agent 2's PBS beliefs (inferred values and confidence) - using visible board for Agent 2
    
    Args:
        actual_board: The actual game board (10x10 tensor)
        agent1_pbs: Agent 1's PBS object (or None if not available)
        agent2_pbs: Agent 2's PBS object (or None if not available)
        episode: Current episode number
        save_path: Path to save the visualization
        visible_board_p1: Visible board for player 1 (shows what player 1 can see)
        visible_board_p2: Visible board for player 2 (shows what player 2 can see)
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'PBS Visualization - Episode {episode}', fontsize=16, fontweight='bold')
    
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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"🎯 PBS visualization saved to {save_path}")


def _plot_actual_board(ax, board_np: np.ndarray, title: str):
    """Plot the actual board with pieces."""
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])  # Remove tick marks
    ax.set_yticks([])  # Remove tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # No grid lines - we'll draw cell borders manually
    
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
            
            # Draw square with cell border (no internal alignment lines)
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


def _plot_pbs_beliefs(ax, visible_board_np: np.ndarray, actual_board_np: np.ndarray, pbs, player_id: int, title: str):
    """
    Plot PBS beliefs for a specific agent.
    
    NOTE: This function uses IDENTICAL logic for both Agent 1 (player_id=1) and Agent 2 (player_id=-1).
    The only difference is in identifying own pieces vs enemy pieces, which is handled by player_id.
    All PBS prediction, color logic, and display logic is the same for both agents.
    """
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])  # Remove tick marks
    ax.set_yticks([])  # Remove tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # No grid lines - we'll draw cell borders manually
    
    # Safety check: ensure PBS object exists and has required attributes
    # SAME CHECK FOR BOTH AGENTS
    if pbs is None:
        ax.text(0.5, 0.5, 'PBS Not Available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return
    
    if not hasattr(pbs, 'belief_distributions'):
        ax.text(0.5, 0.5, 'PBS Not Initialized', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return
    
    # CRITICAL FIX: Iterate through actual board to find ALL enemy pieces
    # This ensures we show PBS beliefs for all enemy pieces, including scouts,
    # even if the visible board isn't correctly initialized with HIDDEN_PIECE
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
                            # Agent 2: enemy pieces are positive (Agent 1's pieces)
                            if actual_piece_val > 0:
                                enemy_positions.add((r, c))
                except (ValueError, TypeError):
                    pass
    
    # Draw board squares with PBS information
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            pos = (r, c)
            # Extract values properly - handle both numpy arrays and tensors
            # Convert to Python scalar for reliable comparison
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
            
            # Extract actual board value - CRITICAL for determining piece ownership
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
            
            # Determine square color and what to display
            # CRITICAL FIX: Use actual board to determine ownership FIRST, not visible board
            # The bug: HIDDEN_PIECE = -3 conflicts with Agent 2's Scout = -3
            # Solution: Check actual_piece_val to determine ownership, not visible_piece_val
            # This way we can distinguish Agent 2's own scout (-3) from HIDDEN_PIECE (-3)
            
            if visible_piece_val == LAKE_SQUARE:
                color = 'lightblue'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                continue
            
            # CRITICAL FIX: Check actual board FIRST to determine if this is an own piece
            # This avoids the HIDDEN_PIECE (-3) vs Agent 2's Scout (-3) conflict
            is_own_piece_by_actual = False
            if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                if player_id == 1:
                    # Agent 1: own pieces are positive in actual board
                    is_own_piece_by_actual = (actual_piece_val > 0)
                elif player_id == -1:
                    # Agent 2: own pieces are negative in actual board
                    is_own_piece_by_actual = (actual_piece_val < 0)
            
            if is_own_piece_by_actual:
                # CRITICAL FIX: This is confirmed to be an own piece using actual board
                # Show it as own piece (green for Agent 2, red for Agent 1)
                color = 'lightcoral' if player_id == 1 else 'lightgreen'  # Red for agent 1, green for agent 2
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                # Show actual piece - use actual_piece_val to get piece type (handles both positive and negative)
                piece_type = PieceType(abs(int(actual_piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                ax.text(c, r, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black')
                continue
            elif visible_piece_val == EMPTY_SQUARE and pos not in enemy_positions:
                # Empty square and not an enemy piece position
                color = 'white'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                continue
            elif visible_piece_val == HIDDEN_PIECE or pos in enemy_positions:
                # CRITICAL: This is a hidden enemy piece - MUST show PBS beliefs
                # For Agent 2, this means Agent 1's pieces (which are hidden)
                # For Agent 1, this means Agent 2's pieces (which are hidden)
                # Hidden enemy piece - show PBS beliefs
                # IDENTICAL LOGIC FOR BOTH AGENT 1 AND AGENT 2
                
                # CRITICAL FIX: Verify this is actually an enemy piece using actual board
                # This prevents showing PBS for empty squares or own pieces
                is_enemy_piece = (pos in enemy_positions)
                
                # CRITICAL FIX: Double-check using actual_piece_val to ensure it's an enemy piece
                # For Agent 2 (player_id == -1): enemy pieces are positive (Agent 1's pieces)
                # For Agent 1 (player_id == 1): enemy pieces are negative (Agent 2's pieces)
                if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                    if player_id == 1:
                        # Agent 1: enemy pieces are negative
                        is_actually_enemy = (actual_piece_val < 0)
                    elif player_id == -1:
                        # Agent 2: enemy pieces are positive (Agent 1's pieces)
                        is_actually_enemy = (actual_piece_val > 0)
                    else:
                        is_actually_enemy = False
                else:
                    is_actually_enemy = False
                
                # CRITICAL FIX: Only show PBS beliefs if this is actually an enemy piece
                # If the actual board shows EMPTY_SQUARE or it's an own piece, skip showing PBS
                if not is_enemy_piece or not is_actually_enemy or actual_piece_val == EMPTY_SQUARE:
                    # This position is empty or not an enemy piece - show as empty, don't show PBS beliefs
                    color = 'white'
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                           facecolor=color, edgecolor='black', linewidth=1.5)
                    ax.add_patch(rect)
                    continue
                
                # CRITICAL FIX: Ensure beliefs are initialized for ALL enemy positions
                # This ensures scouts and all enemy pieces are displayed, even if no actions observed yet
                _ = pbs.belief_distributions[pos]  # Trigger defaultdict to create beliefs if they don't exist
                
                # Use PBS method to get the most likely piece type (with FLAG preference for low confidence)
                revealed_pieces = getattr(pbs, 'revealed_pieces', {})
                if pos not in revealed_pieces:
                    # Get piece type and confidence using PBS method
                    # CRITICAL: NEVER pass actual_piece_val - PBS beliefs should ONLY show predictions, not actual pieces
                    # This ensures PBS beliefs are always based on agent's beliefs, not ground truth
                    # SAME METHOD CALLED FOR BOTH AGENTS - ensures consistent prediction logic
                    piece_type, confidence = _get_pbs_prediction(pbs, pos)
                    piece_rank = PIECE_RANKS.get(piece_type, 0)
                    
                    # Normalize confidence to range from 0.01 (lowest) to 1.0 (highest)
                    confidence_normalized = max(0.01, min(1.0, confidence))
                    
                    # DEBUG: Verify we have the correct actual_piece_val for this enemy piece
                    # For Agent 2's PBS: actual_piece_val should be > 0 (Agent 1's piece)
                    # For Agent 1's PBS: actual_piece_val should be < 0 (Agent 2's piece)
                    if player_id == -1 and actual_piece_val <= 0:
                        # This is a bug - Agent 2's PBS should only show Agent 1's pieces (positive values)
                        # Skip this position to avoid showing wrong color
                        color = 'white'
                        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                               facecolor=color, edgecolor='black', linewidth=1.5)
                        ax.add_patch(rect)
                        continue
                    elif player_id == 1 and actual_piece_val >= 0:
                        # This is a bug - Agent 1's PBS should only show Agent 2's pieces (negative values)
                        # Skip this position to avoid showing wrong color
                        color = 'white'
                        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                               facecolor=color, edgecolor='black', linewidth=1.5)
                        ax.add_patch(rect)
                        continue
                    
                    # Color intensity based on normalized confidence
                    rgba = plt.cm.RdYlGn(confidence_normalized)  # Green (high conf) to Red (low conf)
                    confidence_rgb = np.array(rgba[:3], dtype=np.float64)  # Use RGB only, ensure float64
                else:
                    # Piece is revealed - show actual piece (not prediction)
                    piece_type = None
                    confidence = 0.0
                    confidence_normalized = 0.01
                    piece_rank = 0
                    confidence_rgb = np.array([0.8, 0.8, 0.8], dtype=np.float64)  # Light gray
                
                # Determine base color based on actual piece ownership (for consistency)
                # CRITICAL: For PBS beliefs, we're showing enemy pieces
                # For Agent 2's PBS: showing Agent 1's pieces (enemy) - should be RED
                # For Agent 1's PBS: showing Agent 2's pieces (enemy) - should be GREEN
                # Use actual board to determine which agent's piece this is
                # CRITICAL: actual_piece_val > 0 means Agent 1, < 0 means Agent 2
                # actual_piece_val is already converted to int at the top of the loop
                
                # Determine color based on actual piece ownership
                # CRITICAL FIX: The color MUST match the actual piece ownership from actual_board
                # We've already verified is_actually_enemy above, so we know:
                # - For Agent 2: actual_piece_val > 0 (Agent 1's piece) -> RED
                # - For Agent 1: actual_piece_val < 0 (Agent 2's piece) -> GREEN
                if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                    # CRITICAL: Use actual_piece_val sign to determine color, NOT player_id
                    # actual_piece_val > 0 = Agent 1's piece = RED
                    # actual_piece_val < 0 = Agent 2's piece = GREEN
                    if actual_piece_val > 0:
                        # Agent 1's piece - ALWAYS use red tint (blend red with confidence color)
                        # This is correct for Agent 2's PBS view showing Agent 1's pieces
                        base_rgb = np.array(mcolors.to_rgb('lightcoral'), dtype=np.float64)
                        # Blend: 85% red (ownership), 15% confidence color - very strongly emphasize ownership
                        blended_color = 0.85 * base_rgb + 0.15 * confidence_rgb
                        color = tuple(blended_color)
                    elif actual_piece_val < 0:
                        # Agent 2's piece - ALWAYS use green tint (blend green with confidence color)
                        # This is correct for Agent 1's PBS view showing Agent 2's pieces
                        # CRITICAL: If we reach here for Agent 2's PBS, it's a bug - should have been filtered out
                        base_rgb = np.array(mcolors.to_rgb('lightgreen'), dtype=np.float64)
                        # Blend: 85% green (ownership), 15% confidence color - very strongly emphasize ownership
                        blended_color = 0.85 * base_rgb + 0.15 * confidence_rgb
                        color = tuple(blended_color)
                    else:
                        # Zero or unexpected value, use confidence color only
                        color = tuple(confidence_rgb)
                else:
                    # No actual piece info, use confidence color only
                    color = tuple(confidence_rgb)
                
                # Draw square with cell border (no internal alignment lines)
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                
                # Always show PBS guess for hidden pieces: piece name and confidence (removed ranking)
                # CRITICAL: Show '?' if confidence is below 40% (0.4) - ALWAYS, no exceptions
                # PBS beliefs should ONLY show predictions based on agent's beliefs, never actual pieces
                if piece_type is not None and confidence_normalized >= 0.4:
                    piece_name = PIECE_NAMES.get(piece_type, '?')
                    # Show inferred piece type (centered)
                    ax.text(c, r - 0.15, piece_name, ha='center', va='center', 
                           fontsize=11, fontweight='bold', color='black')
                    # Show confidence score below piece name
                    ax.text(c, r + 0.15, f'C:{confidence_normalized:.2f}', ha='center', va='center', 
                           fontsize=8, color='darkblue', fontweight='bold')
                else:
                    # Show '?' when confidence is too low (< 40%) or piece_type is None
                    ax.text(c, r - 0.15, '?', ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='black')
                    ax.text(c, r + 0.15, f'C:{confidence_normalized:.2f}', ha='center', va='center', 
                           fontsize=8, color='darkblue', fontweight='bold')
            else:
                # Revealed enemy piece (visible_piece_val shows the actual piece)
                # This branch is reached when visible_piece_val is NOT HIDDEN_PIECE and NOT own piece
                # This means the piece was revealed during battle (pieces are only revealed after battles)
                # At setup, all enemy pieces should be HIDDEN_PIECE, so this branch should NOT be reached
                # If this branch is reached at setup, it indicates a bug in board initialization
                # Use actual board to determine ownership for consistency
                # CRITICAL: Use actual_piece_val (not visible_piece_val) to determine ownership
                # CRITICAL: This shows the ACTUAL revealed piece, NOT a prediction
                if actual_piece_val is not None and actual_piece_val != EMPTY_SQUARE and actual_piece_val != LAKE_SQUARE:
                    if actual_piece_val > 0:
                        # This is agent 1's piece - always red
                        color = 'lightcoral'
                    elif actual_piece_val < 0:
                        # This is agent 2's piece - always green
                        color = 'lightgreen'
                    else:
                        # Fallback to visible_piece_val if actual_piece_val is 0
                        color = 'lightcoral' if visible_piece_val > 0 else 'lightgreen'
                else:
                    # Fallback: use visible_piece_val to determine color
                    color = 'lightcoral' if visible_piece_val > 0 else 'lightgreen'
                
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                # Show revealed piece (removed ranking display)
                piece_type = PieceType(abs(int(visible_piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                # Show piece name (centered)
                ax.text(c, r - 0.15, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black')
                # Show confidence (1.0 for revealed pieces since they're known)
                ax.text(c, r + 0.15, f'C:1.00', ha='center', va='center', 
                       fontsize=8, color='darkblue', fontweight='bold')


def _get_pbs_prediction(pbs, pos: tuple) -> tuple:
    """
    Get PBS prediction for a position based ONLY on beliefs, never on actual pieces.
    
    CRITICAL: PBS beliefs should ONLY show what the agent believes, NOT what actually is.
    This function should NEVER use actual board values - only the agent's beliefs.
    
    Args:
        pbs: ProbabilisticBeliefState object
        pos: Position tuple (r, c)
        
    Returns:
        Tuple of (piece_type, confidence)
    """
    uniform_prob = 1.0 / len(PieceType)
    
    # Get beliefs if they exist
    if pos in pbs.belief_distributions:
        beliefs = pbs.belief_distributions[pos]
        belief_values = list(beliefs.values())
        
        if len(belief_values) > 0:
            min_val = min(belief_values)
            max_val = max(belief_values)
            # Consider uniform if difference is very small (within floating point tolerance)
            is_uniform = (max_val - min_val) < 1e-6
            
            if is_uniform:
                # When beliefs are uniform (setup time), return FLAG as default
                # This is correct - at setup, agent doesn't know what pieces are, so shows FLAG
                # The visualization will show '?' because confidence is below 40%
                return PieceType.FLAG, uniform_prob
            else:
                # Find most likely piece type from existing beliefs
                sorted_beliefs = sorted(beliefs.items(), key=lambda x: (-x[1], x[0].value))
                piece_type, confidence = sorted_beliefs[0]
                
                # If confidence is very low (close to uniform), use FLAG as default
                if confidence <= uniform_prob * 1.15:  # Within 15% of uniform
                    return PieceType.FLAG, uniform_prob
                # Also ensure FLAG is always preferred when confidence is very low
                elif piece_type == PieceType.SPY and confidence < uniform_prob * 1.5:
                    # If SPY is selected but confidence is still low, prefer FLAG
                    return PieceType.FLAG, uniform_prob
                else:
                    return piece_type, confidence
        else:
            # Empty beliefs - use FLAG as default
            return PieceType.FLAG, uniform_prob
    else:
        # No beliefs exist - use FLAG as default
        return PieceType.FLAG, uniform_prob

