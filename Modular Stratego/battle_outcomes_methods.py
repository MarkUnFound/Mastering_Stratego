# Battle Outcome Methods for KLUSS
# Insert after apply_action method in kluss_solver.py

def get_battle_outcomes(self, state, action):
    """
    STRATEGO SPECIFIC: Returns list of (probability, next_state) for battle outcomes.
    If attacking an unknown piece, branch based on belief probabilities.
    
    Returns:
        List[Tuple[float, state]]: [(prob1, state1), (prob2, state2), ...]
    """
    (r1, c1), (r2, c2) = action
    board = state.board
    if hasattr(board, 'cpu'): board = board.cpu().numpy()
    
    attacker_val = board[r1, c1]
    if hasattr(attacker_val, 'item'): attacker_val = attacker_val.item()
    
    defender_val = board[r2, c2]
    if hasattr(defender_val, 'item'): defender_val = defender_val.item()
    
    # Case 1: Moving to empty square (deterministic)
    if defender_val == 0 or defender_val == LAKE_SQUARE:
        next_state = self._apply_deterministic_move(state, action)
        return [(1.0, next_state)]
    
    # Case 2: Battle
    attacker_owner = 1 if attacker_val > 0 else -1
    defender_owner = 1 if defender_val > 0 else -1
    
    # Check if it's actually an enemy piece
    if attacker_owner == defender_owner:
        # Same team (shouldn't happen with proper move generation)
        return []
    
    # Query belief state for defender probabilities
    if self.belief_state and hasattr(self.belief_state, 'get_piece_probabilities'):
        defender_probs = self.belief_state.get_piece_probabilities(state, (r2, c2))
    else:
        # Fallback: assume uniform or use single revealed value
        defender_probs = {abs(defender_val): 1.0}
    
    # Calculate outcome probabilities
    attacker_type = abs(attacker_val)
    outcomes = []
    
    # Aggregate outcomes by result (win/loss/draw)
    win_prob = 0.0
    loss_prob = 0.0
    draw_prob = 0.0
    
    for defender_type, type_prob in defender_probs.items():
        if type_prob < 0.01:  # Skip negligible probabilities
            continue
        
        # Determine battle outcome based on Stratego rules
        result = self._resolve_battle(attacker_type, defender_type)
        
        if result == 1:  # Attacker wins
            win_prob += type_prob
        elif result == -1:  # Defender wins
            loss_prob += type_prob
        else:  # Draw
            draw_prob += type_prob
    
    # Create state for each unique outcome
    if win_prob > 0:
        win_state = self._apply_battle_win(state, action)
        outcomes.append((win_prob, win_state))
    
    if loss_prob > 0:
        loss_state = self._apply_battle_loss(state, action)
        outcomes.append((loss_prob, loss_state))
    
    if draw_prob > 0:
        draw_state = self._apply_battle_draw(state, action)
        outcomes.append((draw_prob, draw_state))
    
    return outcomes if outcomes else [(1.0, self._apply_deterministic_move(state, action))]

def _resolve_battle(self, attacker_type, defender_type):
    """
    Determine battle outcome based on Stratego rules.
    Returns: 1 (attacker wins), -1 (defender wins), 0 (draw)
    """
    from piece import PieceType
    
    # Special cases
    if defender_type == PieceType.BOMB.value:
        return 1 if attacker_type == PieceType.MINER.value else -1
    
    if defender_type == PieceType.FLAG.value:
        return 1  # Attacker always wins (captures flag)
    
    if attacker_type == PieceType.SPY.value and defender_type == PieceType.MARSHAL.value:
        return 1  # Spy kills Marshal
    
    # Standard ranking comparison
    if attacker_type > defender_type:
        return 1  # Higher rank wins
    elif attacker_type < defender_type:
        return -1  # Lower rank loses
    else:
        return 0  # Equal ranks = mutual destruction

def _apply_deterministic_move(self, state, action):
    """Move piece to empty square."""
    new_state = copy.deepcopy(state)
    (r1, c1), (r2, c2) = action
    new_state.board[r1, c1] = 0
    new_state.board[r2, c2] = state.board[r1, c1]
    new_state.current_player *= -1
    return new_state

def _apply_battle_win(self, state, action):
    """Attacker wins battle."""
    new_state = copy.deepcopy(state)
    (r1, c1), (r2, c2) = action
    new_state.board[r2, c2] = state.board[r1, c1]  # Attacker takes square
    new_state.board[r1, c1] = 0
    new_state.current_player *= -1
    return new_state

def _apply_battle_loss(self, state, action):
    """Defender wins battle."""
    new_state = copy.deepcopy(state)
    (r1, c1), (r2, c2) = action
    new_state.board[r1, c1] = 0  # Attacker dies, defender stays
    new_state.current_player *= -1
    return new_state

def _apply_battle_draw(self, state, action):
    """Both pieces destroyed."""
    new_state = copy.deepcopy(state)
    (r1, c1), (r2, c2) = action
    new_state.board[r1, c1] = 0
    new_state.board[r2, c2] = 0
    new_state.current_player *= -1
    return new_state
