"""
Setup evaluation and reward calculation functions.
"""
from typing import List, Tuple, Optional
import random
from piece import PieceType, PIECE_RANKS

def evaluate_flag_protection(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                            player_id: int, board_size: int = 10) -> float:
    """
    Evaluate how well the flag is protected by bombs, lakes, or strong pieces.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        board_size: Size of the board (default 10)
        
    Returns:
        Protection score (0.0 to 1.0, higher is better)
    """
    # Find flag position
    flag_pos = None
    for piece, pos in placement:
        if piece == PieceType.FLAG:
            flag_pos = pos
            break
    
    if flag_pos is None:
        return 0.0  # No flag found
    
    flag_r, flag_c = flag_pos
    
    # Define adjacent positions (4 directions: up, down, left, right)
    adjacent_positions = [
        (flag_r - 1, flag_c),  # Up
        (flag_r + 1, flag_c),  # Down
        (flag_r, flag_c - 1),  # Left
        (flag_r, flag_c + 1),  # Right
    ]
    
    # Filter valid positions (within board bounds)
    valid_adjacent = [(r, c) for r, c in adjacent_positions 
                     if 0 <= r < board_size and 0 <= c < board_size]
    
    # Create a map of positions to pieces
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Check for protection
    protection_score = 0.0
    
    # IMPROVED: Add row-based vulnerability penalty
    # Flags closer to enemy (front rows) are more vulnerable
    if player_id == 1:
        # Player 1: row 9 is front (most vulnerable), row 6 is back (safest)
        # Vulnerability: 0.0 (back) to 1.0 (front)
        row_vulnerability = (flag_r - 6) / 3.0 if flag_r >= 6 else 0.0
    else:  # player_id == -1
        # Player 2: row 0 is front (most vulnerable), row 3 is back (safest)
        # Vulnerability: 0.0 (back) to 1.0 (front)
        row_vulnerability = (3 - flag_r) / 3.0 if flag_r <= 3 else 0.0
    
    # Reduce protection score based on vulnerability
    # Front-row flags need MUCH more protection to get same score
    vulnerability_penalty = row_vulnerability * 0.5  # Reduce protection score by up to 50% for front-row flags
    
    for adj_pos in valid_adjacent:
        if adj_pos in position_to_piece:
            piece = position_to_piece[adj_pos]
            
            # Check if it's a bomb (strong protection)
            if piece == PieceType.BOMB:
                protection_score += 0.4  # Bomb provides strong protection
            # Check if it's a strong piece (MARSHAL, GENERAL, COLONEL)
            elif piece in [PieceType.MARSHAL, PieceType.GENERAL, PieceType.COLONEL]:
                protection_score += 0.3  # Strong pieces provide good protection
            # Check if it's a medium piece (MAJOR, CAPTAIN)
            elif piece in [PieceType.MAJOR, PieceType.CAPTAIN]:
                protection_score += 0.2  # Medium pieces provide some protection
            # Any other piece provides minimal protection
            else:
                protection_score += 0.1
        else:
            # Check if it's a lake (lakes provide protection by blocking movement)
            # Lakes are at fixed positions: (4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)
            lakes = [(4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)]
            if adj_pos in lakes:
                protection_score += 0.3  # Lake provides protection
    
    # Apply vulnerability penalty (front-row flags need more protection)
    protection_score = max(0.0, protection_score - vulnerability_penalty)
    
    # Normalize score (max possible is 4 adjacent positions * 0.4 = 1.6, but we cap at 1.0)
    return min(1.0, protection_score)


def evaluate_piece_distribution(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                player_id: int) -> float:
    """
    Evaluate balanced piece distribution - reward spreading strong pieces across rows.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Distribution score (0.0 to 1.0, higher is better)
    """
    # Define strong pieces
    strong_pieces = [PieceType.MARSHAL, PieceType.GENERAL, PieceType.COLONEL, 
                     PieceType.MAJOR, PieceType.CAPTAIN]
    
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]  # Bottom rows
    else:
        player_rows = [0, 1, 2, 3]  # Top rows
    
    # Count strong pieces per row
    row_counts = {row: 0 for row in player_rows}
    total_strong = 0
    
    for piece, (r, c) in placement:
        if piece in strong_pieces and r in player_rows:
            row_counts[r] += 1
            total_strong += 1
    
    if total_strong == 0:
        return 0.5  # Neutral if no strong pieces
    
    # Calculate distribution variance (lower variance = better distribution)
    counts = list(row_counts.values())
    mean_count = sum(counts) / len(counts)
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    
    # Normalize: perfect distribution (all rows equal) = 1.0, all in one row = 0.0
    max_variance = (total_strong ** 2) / len(player_rows)  # Worst case: all in one row
    if max_variance == 0:
        return 1.0
    
    distribution_score = 1.0 - (variance / max_variance)
    return max(0.0, min(1.0, distribution_score))


def evaluate_scout_placement(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                            player_id: int) -> float:
    """
    Reward scouts placed in forward positions for early scouting.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Scout placement score (0.0 to 1.0, higher is better)
    """
    # Forward rows for each player (closest to enemy)
    if player_id == 1:
        forward_rows = [6]  # Row 6 is closest to enemy (row 5 is lakes, row 4 is enemy territory)
    else:
        forward_rows = [3]  # Row 3 is closest to enemy (row 4 is lakes, row 5 is enemy territory)
    
    scouts = [(piece, pos) for piece, pos in placement if piece == PieceType.SCOUT]
    if len(scouts) == 0:
        return 0.0
    
    # Count scouts in forward positions
    forward_scouts = sum(1 for _, (r, c) in scouts if r in forward_rows)
    
    # Reward: 0.5 base + 0.5 for forward placement ratio
    forward_ratio = forward_scouts / len(scouts)
    return 0.5 + (0.5 * forward_ratio)


def evaluate_bomb_placement(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                           player_id: int) -> float:
    """
    Reward bombs protecting key pieces (not just flag).
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Bomb placement score (0.0 to 1.0, higher is better)
    """
    # Create position to piece map
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Key pieces that should be protected (high value pieces)
    key_pieces = [PieceType.FLAG, PieceType.MARSHAL, PieceType.GENERAL]
    
    # Find key piece positions
    key_positions = [pos for piece, pos in placement if piece in key_pieces]
    
    if len(key_positions) == 0:
        return 0.0
    
    # Count how many key pieces have bombs adjacent
    protected_count = 0
    
    for key_pos in key_positions:
        key_r, key_c = key_pos
        adjacent = [
            (key_r - 1, key_c), (key_r + 1, key_c),
            (key_r, key_c - 1), (key_r, key_c + 1)
        ]
        
        # Check if any adjacent position has a bomb
        for adj_pos in adjacent:
            if adj_pos in position_to_piece:
                if position_to_piece[adj_pos] == PieceType.BOMB:
                    protected_count += 1
                    break  # Count each key piece only once
    
    # Score: ratio of protected key pieces
    return protected_count / len(key_positions)


def evaluate_defensive_formation(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                 player_id: int) -> float:
    """
    Reward pieces forming defensive lines (pieces in same row/column).
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Formation score (0.0 to 1.0, higher is better)
    """
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]
    else:
        player_rows = [0, 1, 2, 3]
    
    # Count pieces per row and column
    row_counts = {r: 0 for r in player_rows}
    col_counts = {c: 0 for c in range(10)}
    
    for piece, (r, c) in placement:
        if r in player_rows:
            row_counts[r] += 1
            col_counts[c] += 1
    
    # Reward rows with multiple pieces (defensive lines)
    row_score = sum(1 for count in row_counts.values() if count >= 8) / len(player_rows)
    
    # Reward columns with multiple pieces (vertical defense)
    col_score = sum(1 for count in col_counts.values() if count >= 3) / 10.0
    
    # Average of row and column formation
    return (row_score + col_score) / 2.0


def evaluate_piece_coordination(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                player_id: int) -> float:
    """
    Reward pieces that can support each other (e.g., miners near bombs).
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Coordination score (0.0 to 1.0, higher is better)
    """
    # Create position to piece map
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Find bomb positions
    bomb_positions = [pos for piece, pos in placement if piece == PieceType.BOMB]
    
    if len(bomb_positions) == 0:
        return 0.0
    
    # Count miners adjacent to bombs (miners can defuse bombs)
    coordinated_count = 0
    
    for bomb_pos in bomb_positions:
        bomb_r, bomb_c = bomb_pos
        adjacent = [
            (bomb_r - 1, bomb_c), (bomb_r + 1, bomb_c),
            (bomb_r, bomb_c - 1), (bomb_r, bomb_c + 1)
        ]
        
        # Check if any adjacent position has a miner
        for adj_pos in adjacent:
            if adj_pos in position_to_piece:
                if position_to_piece[adj_pos] == PieceType.MINER:
                    coordinated_count += 1
                    break  # Count each bomb only once
    
    # Score: ratio of bombs with adjacent miners
    return coordinated_count / len(bomb_positions)


def evaluate_piece_value_distribution(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                     player_id: int) -> float:
    """
    Reward for spreading high-value pieces across rows (not all in one row).
    Prevents clustering of strong pieces.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Distribution score (0.0 to 1.0, higher is better)
    """
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]
    else:
        player_rows = [0, 1, 2, 3]
    
    # Calculate total piece value per row
    row_values = {r: 0.0 for r in player_rows}
    
    for piece, (r, c) in placement:
        if r in player_rows:
            piece_value = PIECE_RANKS.get(piece, 0)
            row_values[r] += piece_value
    
    # Calculate variance (lower variance = better distribution)
    values = [v for v in row_values.values() if v > 0]
    if len(values) == 0:
        return 0.0
    
    mean_value = sum(values) / len(values)
    if mean_value == 0:
        return 0.0
    
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    max_variance = mean_value ** 2  # Worst case: all value in one row
    
    if max_variance == 0:
        return 1.0
    
    distribution_score = 1.0 - (variance / max_variance)
    return max(0.0, min(1.0, distribution_score))


def evaluate_strategic_positioning(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                  player_id: int) -> float:
    """
    Reward for strategic piece positioning:
    - High-value pieces in center/back (protected)
    - Scouts in front (aggressive)
    - Bombs near flag (defensive)
    - Miners near bombs (tactical)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Strategic positioning score (0.0 to 1.0, higher is better)
    """
    score = 0.0
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Find flag position
    flag_pos = next((pos for piece, pos in placement if piece == PieceType.FLAG), None)
    
    for piece, (r, c) in placement:
        piece_value = PIECE_RANKS.get(piece, 0)
        
        # High-value pieces (8+) should be in back rows
        if piece_value >= 8:
            if player_id == 1:
                if r >= 7:  # Back rows (7-9)
                    score += 0.1
            else:
                if r <= 2:  # Back rows (0-2)
                    score += 0.1
        
        # Scouts should be in front rows
        if piece == PieceType.SCOUT:
            if player_id == 1:
                if r >= 8:  # Front rows (8-9)
                    score += 0.05
            else:
                if r <= 1:  # Front rows (0-1)
                    score += 0.05
        
        # Bombs should be near flag
        if piece == PieceType.BOMB and flag_pos:
            flag_r, flag_c = flag_pos
            distance = abs(r - flag_r) + abs(c - flag_c)
            if distance <= 2:
                score += 0.1
        
        # Miners should be near bombs
        if piece == PieceType.MINER:
            for bomb_pos, bomb_piece in position_to_piece.items():
                if bomb_piece == PieceType.BOMB:
                    bomb_r, bomb_c = bomb_pos
                    distance = abs(r - bomb_r) + abs(c - bomb_c)
                    if distance <= 2:
                        score += 0.05
                        break
    
    return min(1.0, score)


def evaluate_defensive_depth(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                            player_id: int) -> float:
    """
    Reward for creating multiple defensive layers (not just one row).
    Strong pieces in multiple rows provide better defense.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Defensive depth score (0.0 to 1.0, higher is better)
    """
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]
    else:
        player_rows = [0, 1, 2, 3]
    
    # Count strong pieces (value >= 7) per row
    row_strong_pieces = {r: 0 for r in player_rows}
    
    for piece, (r, c) in placement:
        if r in player_rows:
            piece_value = PIECE_RANKS.get(piece, 0)
            if piece_value >= 7:
                row_strong_pieces[r] += 1
    
    # Reward for having strong pieces in multiple rows
    rows_with_strong = sum(1 for count in row_strong_pieces.values() if count > 0)
    
    if rows_with_strong >= 3:
        return 1.0
    elif rows_with_strong == 2:
        return 0.6
    elif rows_with_strong == 1:
        return 0.3
    else:
        return 0.0


def evaluate_piece_synergy(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                           player_id: int) -> float:
    """
    Reward for placing pieces that synergize:
    - Marshal/General near each other (command structure)
    - Miners near bombs (defusing capability)
    - Strong pieces protecting weaker ones
    - Scouts in groups (coordination)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Synergy score (0.0 to 1.0, higher is better)
    """
    score = 0.0
    position_to_piece = {pos: piece for piece, pos in placement}
    
    for piece, (r, c) in placement:
        # Check adjacent pieces for synergy
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = r + dr, c + dc
                adj_pos = (adj_r, adj_c)
                
                if adj_pos in position_to_piece:
                    adj_piece = position_to_piece[adj_pos]
                    
                    # Marshal/General synergy
                    if piece in [PieceType.MARSHAL, PieceType.GENERAL]:
                        if adj_piece in [PieceType.MARSHAL, PieceType.GENERAL, PieceType.COLONEL]:
                            score += 0.05
                    
                    # Miner-Bomb synergy
                    if piece == PieceType.MINER and adj_piece == PieceType.BOMB:
                        score += 0.1
                    
                    # Strong-weak protection
                    piece_value = PIECE_RANKS.get(piece, 0)
                    adj_value = PIECE_RANKS.get(adj_piece, 0)
                    if piece_value >= 8 and adj_value < 5:
                        score += 0.03
    
    return min(1.0, score)


def evaluate_vulnerability(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                          player_id: int) -> float:
    """
    Penalty for vulnerable piece placements:
    - High-value pieces in front rows (exposed)
    - Flag with weak protection
    - Isolated pieces (no support)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Vulnerability penalty score (0.0 to 1.0, higher = more vulnerable)
    """
    penalty = 0.0
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Find flag
    flag_pos = next((pos for piece, pos in placement if piece == PieceType.FLAG), None)
    
    for piece, (r, c) in placement:
        piece_value = PIECE_RANKS.get(piece, 0)
        
        # High-value pieces in front rows
        if piece_value >= 9:
            if player_id == 1:
                if r >= 8:  # Front row (8-9)
                    penalty += 0.2
            else:
                if r <= 1:  # Front row (0-1)
                    penalty += 0.2
        
        # Isolated pieces (no adjacent friendly pieces)
        adjacent_friendly = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = r + dr, c + dc
                adj_pos = (adj_r, adj_c)
                if adj_pos in position_to_piece:
                    adjacent_friendly += 1
        
        if adjacent_friendly == 0 and piece_value >= 7:
            penalty += 0.1
    
    # Flag vulnerability
    if flag_pos:
        flag_r, flag_c = flag_pos
        protection_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = flag_r + dr, flag_c + dc
                if (adj_r, adj_c) in position_to_piece:
                    protection_count += 1
        
        if protection_count < 2:
            penalty += 0.3
    
    return min(1.0, penalty)


def calculate_setup_agent_reward(placement: List[Tuple[PieceType, Tuple[int, int]]],
                                  player_id: int,
                                  winner: Optional[int],
                                  move_count: int,
                                  min_survival_moves: int = 100) -> float:
    """
    Calculate reward for setup agent based on:
    1. Flag protection (bombs, lakes, strong pieces)
    2. Game length (penalty for short games, reward for long games)
    3. Win/loss outcome
    4. Piece distribution (balanced placement)
    5. Scout placement (forward positions)
    6. Bomb placement (protecting key pieces)
    7. Defensive formation (pieces in lines)
    8. Piece coordination (miners near bombs)
    9. Early game survival (flag survives first 50 moves)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        winner: Winner of the game (1, -1, or None for draw)
        move_count: Number of moves in the game
        min_survival_moves: Minimum moves to avoid penalty (default 100)
        
    Returns:
        Total reward for the setup agent
    """
    reward = 0.0
    
    # Find flag position for row-based penalties
    flag_pos = None
    for piece, pos in placement:
        if piece == PieceType.FLAG:
            flag_pos = pos
            break
    
    # 0. CRITICAL: Penalty for flag in front row (very vulnerable)
    # ADJUSTED: Reduced penalty to allow learning while still discouraging bad placement
    if flag_pos is not None:
        flag_r, flag_c = flag_pos
        # Player 1's front row is row 9 (closest to enemy)
        # Player 2's front row is row 0 (closest to enemy)
        if (player_id == 1 and flag_r == 9) or (player_id == -1 and flag_r == 0):
            reward -= 0.5  # Scaled down by 10x (was -5.0, now -0.5)
        # Penalty for second row (still vulnerable)
        elif (player_id == 1 and flag_r == 8) or (player_id == -1 and flag_r == 1):
            reward -= 0.25  # Scaled down by 10x (was -2.5, now -0.25)
        # Bonus for back rows (safer)
        elif (player_id == 1 and flag_r <= 6) or (player_id == -1 and flag_r >= 3):
            reward += 0.3  # Scaled down by 10x (was 3.0, now 0.3)
    
    # 1. Flag protection reward (0.0 to 1.0, scaled to 0-0.5)
    protection_score = evaluate_flag_protection(placement, player_id)
    reward += protection_score * 0.5  # Scaled down by 10x (was 5.0, now 0.5)
    
    # 2. Game length reward/penalty (scaled down by 10x)
    if move_count < min_survival_moves:
        # Penalty for short games (games that end too quickly)
        # Linear penalty: -0.01 per move below threshold (was -0.1)
        penalty = -0.01 * (min_survival_moves - move_count)
        reward += penalty
    else:
        # Reward for surviving longer (games that last at least min_survival_moves)
        # Small reward for each move above threshold
        bonus = 0.001 * (move_count - min_survival_moves)  # Scaled down by 10x (was 0.01)
        reward += bonus
    
    # 3. Win/loss reward (scaled down by 10x)
    if winner == player_id:
        # Big reward for winning
        reward += 1.0  # Scaled down by 10x (was 10.0)
    elif winner is not None and winner != player_id:
        # Penalty for losing (but less severe than short game penalty)
        reward -= 0.2  # Scaled down by 10x (was -2.0)
    else:
        # Small reward for draw
        reward += 0.1  # Scaled down by 10x (was 1.0)
    
    # 4. Piece distribution bonus (0.0 to 1.0, scaled to 0-0.2)
    distribution_score = evaluate_piece_distribution(placement, player_id)
    reward += distribution_score * 0.2  # Scaled down by 10x (was 2.0)
    
    # 5. Scout placement reward (0.0 to 1.0, scaled to 0-0.15)
    scout_score = evaluate_scout_placement(placement, player_id)
    reward += scout_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 6. Bomb placement reward (0.0 to 1.0, scaled to 0-0.2)
    bomb_score = evaluate_bomb_placement(placement, player_id)
    reward += bomb_score * 0.2  # Scaled down by 10x (was 2.0)
    
    # 7. Defensive formation reward (0.0 to 1.0, scaled to 0-0.15)
    formation_score = evaluate_defensive_formation(placement, player_id)
    reward += formation_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 8. Piece coordination reward (0.0 to 1.0, scaled to 0-0.1)
    coordination_score = evaluate_piece_coordination(placement, player_id)
    reward += coordination_score * 0.1  # Scaled down by 10x (was 1.0)
    
    # 9. Early game survival bonus (extra reward if flag survives first 50 moves)
    if move_count >= 50:
        # Bonus for surviving early game
        reward += 0.2  # Scaled down by 10x (was 2.0)
    
    # 1. Piece Value Distribution Rewards (0.0 to 1.0, scaled to 0-0.2)
    value_distribution_score = evaluate_piece_value_distribution(placement, player_id)
    reward += value_distribution_score * 0.2  # Scaled down by 10x (was 2.0)
    
    # 2. Strategic Piece Positioning (0.0 to 1.0, scaled to 0-0.25)
    strategic_score = evaluate_strategic_positioning(placement, player_id)
    reward += strategic_score * 0.25  # Scaled down by 10x (was 2.5)
    
    # 3. Defensive Depth Rewards (0.0 to 1.0, scaled to 0-0.15)
    defensive_depth_score = evaluate_defensive_depth(placement, player_id)
    reward += defensive_depth_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 4. Piece Synergy Rewards (0.0 to 1.0, scaled to 0-0.15)
    synergy_score = evaluate_piece_synergy(placement, player_id)
    reward += synergy_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 5. Vulnerability Assessment Penalties (0.0 to 1.0, scaled to 0-0.3 penalty)
    vulnerability_penalty = evaluate_vulnerability(placement, player_id)
    reward -= vulnerability_penalty * 0.3  # Scaled down by 10x (was 3.0)
    
    return reward

def generate_heuristic_placement(player_id: int) -> List[Tuple[PieceType, Tuple[int, int]]]:
    """
    Generate a heuristic placement:
    - Weakest pieces at the back (Flag, Spy, Scouts, Miners)
    - Strongest pieces at the front (Marshal, General, Colonels, Majors, Bombs)
    """
    # Define piece groups
    back_pieces = [PieceType.FLAG, PieceType.SPY] + [PieceType.SCOUT]*8 + [PieceType.MINER]*5 + [PieceType.SERGEANT]*4
    front_pieces = [PieceType.MARSHAL, PieceType.GENERAL] + [PieceType.COLONEL]*2 + [PieceType.MAJOR]*3 + \
                   [PieceType.CAPTAIN]*4 + [PieceType.LIEUTENANT]*4 + [PieceType.BOMB]*6
    
    random.shuffle(back_pieces)
    random.shuffle(front_pieces)
    
    placement = []
    
    if player_id == 1:
        rows_back = [(r, c) for r in range(8, 10) for c in range(10)]
        rows_front = [(r, c) for r in range(6, 8) for c in range(10)]
        
        for pos in rows_back:
            if back_pieces:
                piece = back_pieces.pop()
            else:
                piece = front_pieces.pop()
            placement.append((piece, pos))
            
        for pos in rows_front:
            if front_pieces:
                piece = front_pieces.pop()
            else:
                piece = back_pieces.pop() 
            placement.append((piece, pos))
            
    else:
        rows_back = [(r, c) for r in range(0, 2) for c in range(10)]
        rows_front = [(r, c) for r in range(2, 4) for c in range(10)]
        
        for pos in rows_back:
            if back_pieces:
                piece = back_pieces.pop()
            else:
                piece = front_pieces.pop()
            placement.append((piece, pos))
            
        for pos in rows_front:
            if front_pieces:
                piece = front_pieces.pop()
            else:
                piece = back_pieces.pop()
            placement.append((piece, pos))
            
    return placement
