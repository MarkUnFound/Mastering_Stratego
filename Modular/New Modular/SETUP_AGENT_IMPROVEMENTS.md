# Setup Agent Improvements and Rewards

## Current Setup Agent Reward Structure
- ✅ Flag placement penalties/bonuses (front row: -5.0, back rows: +3.0)
- ✅ Flag protection (0-5.0)
- ✅ Game length rewards/penalties
- ✅ Win/loss rewards (+10.0 win, -2.0 loss)
- ✅ Piece distribution (0-2.0)
- ✅ Scout placement (0-1.5)
- ✅ Bomb placement (0-2.0)
- ✅ Defensive formation (0-1.5)
- ✅ Piece coordination (0-1.0)
- ✅ Early game survival (+2.0)

## Recommended Additional Rewards and Improvements

### 1. **Piece Value Distribution Rewards** (High Priority)
**Proposed:** Reward for balanced piece value distribution across rows

```python
def evaluate_piece_value_distribution(placement, player_id):
    """
    Reward for spreading high-value pieces across rows (not all in one row).
    Prevents clustering of strong pieces.
    """
    # Calculate total piece value per row
    row_values = defaultdict(float)
    for piece, (r, c) in placement:
        piece_value = PIECE_RANKS.get(piece, 0)
        row_values[r] += piece_value
    
    # Calculate variance (lower variance = better distribution)
    values = list(row_values.values())
    if len(values) == 0:
        return 0.0
    
    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    max_variance = mean_value ** 2  # Worst case: all value in one row
    
    if max_variance == 0:
        return 1.0
    
    distribution_score = 1.0 - (variance / max_variance)
    return max(0.0, min(1.0, distribution_score))

# Reward: 0.0 to 1.0, scaled to 0-2.0
value_distribution_score = evaluate_piece_value_distribution(placement, player_id)
reward += value_distribution_score * 2.0
```

**Benefit:** Encourages balanced piece placement, prevents weak spots

---

### 2. **Strategic Piece Positioning** (High Priority)
**Proposed:** Reward for placing pieces in strategic positions

```python
def evaluate_strategic_positioning(placement, player_id):
    """
    Reward for strategic piece positioning:
    - High-value pieces in center/back (protected)
    - Scouts in front (aggressive)
    - Bombs near flag (defensive)
    - Miners near bombs (tactical)
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
                if r >= 7:  # Back rows
                    score += 0.1
            else:
                if r <= 2:  # Back rows
                    score += 0.1
        
        # Scouts should be in front rows
        if piece == PieceType.SCOUT:
            if player_id == 1:
                if r >= 8:  # Front rows
                    score += 0.05
            else:
                if r <= 1:  # Front rows
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

# Reward: 0.0 to 1.0, scaled to 0-2.5
strategic_score = evaluate_strategic_positioning(placement, player_id)
reward += strategic_score * 2.5
```

**Benefit:** Encourages strategic piece placement based on piece roles

---

### 3. **Defensive Depth Rewards** (High Priority)
**Proposed:** Reward for creating defensive layers

```python
def evaluate_defensive_depth(placement, player_id):
    """
    Reward for creating multiple defensive layers (not just one row).
    Strong pieces in multiple rows provide better defense.
    """
    # Count strong pieces (value >= 7) per row
    row_strong_pieces = defaultdict(int)
    for piece, (r, c) in placement:
        if PIECE_RANKS.get(piece, 0) >= 7:
            row_strong_pieces[r] += 1
    
    # Reward for having strong pieces in multiple rows
    rows_with_strong = len(row_strong_pieces)
    if rows_with_strong >= 3:
        return 1.0
    elif rows_with_strong == 2:
        return 0.6
    elif rows_with_strong == 1:
        return 0.3
    else:
        return 0.0

# Reward: 0.0 to 1.0, scaled to 0-1.5
defensive_depth_score = evaluate_defensive_depth(placement, player_id)
reward += defensive_depth_score * 1.5
```

**Benefit:** Encourages layered defense, not just one strong row

---

### 4. **Piece Synergy Rewards** (Medium Priority)
**Proposed:** Reward for pieces that work well together

```python
def evaluate_piece_synergy(placement, player_id):
    """
    Reward for placing pieces that synergize:
    - Marshal/General near each other (command structure)
    - Miners near bombs (defusing capability)
    - Strong pieces protecting weaker ones
    - Scouts in groups (coordination)
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

# Reward: 0.0 to 1.0, scaled to 0-1.5
synergy_score = evaluate_piece_synergy(placement, player_id)
reward += synergy_score * 1.5
```

**Benefit:** Encourages tactical piece relationships

---

### 5. **Vulnerability Assessment Penalties** (Medium Priority)
**Proposed:** Penalty for leaving pieces vulnerable

```python
def evaluate_vulnerability(placement, player_id):
    """
    Penalty for vulnerable piece placements:
    - High-value pieces in front rows (exposed)
    - Flag with weak protection
    - Isolated pieces (no support)
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
                if r >= 8:  # Front row
                    penalty += 0.2
            else:
                if r <= 1:  # Front row
                    penalty += 0.2
        
        # Isolated pieces (no adjacent friendly pieces)
        adjacent_friendly = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = r + dr, c + dc
                if (adj_r, adj_c) in position_to_piece:
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

# Penalty: 0.0 to 1.0, scaled to 0-3.0
vulnerability_penalty = evaluate_vulnerability(placement, player_id)
reward -= vulnerability_penalty * 3.0
```

**Benefit:** Prevents vulnerable placements

---

### 6. **Center Control Preparation** (Medium Priority)
**Proposed:** Reward for pieces positioned to control center

```python
def evaluate_center_control_prep(placement, player_id):
    """
    Reward for placing pieces that can quickly control center (rows 4-5).
    Scouts and mobile pieces in forward positions.
    """
    score = 0.0
    
    for piece, (r, c) in placement:
        # Calculate distance to center
        center_rows = [4, 5]
        min_dist_to_center = min(abs(r - cr) for cr in center_rows)
        
        # Scouts in forward positions (can reach center quickly)
        if piece == PieceType.SCOUT:
            if player_id == 1:
                if r >= 7:  # Forward position
                    score += 0.05
            else:
                if r <= 2:  # Forward position
                    score += 0.05
        
        # Mobile pieces (not bombs/flags) near center
        if piece not in [PieceType.BOMB, PieceType.FLAG]:
            if min_dist_to_center <= 2:
                score += 0.02
    
    return min(1.0, score)

# Reward: 0.0 to 1.0, scaled to 0-1.0
center_control_score = evaluate_center_control_prep(placement, player_id)
reward += center_control_score * 1.0
```

**Benefit:** Encourages strategic positioning for mid-game control

---

### 7. **Piece Economy Rewards** (Low Priority)
**Proposed:** Reward for efficient piece placement

```python
def evaluate_piece_economy(placement, player_id):
    """
    Reward for efficient piece placement:
    - No wasted positions (all pieces placed optimally)
    - Strong pieces in valuable positions
    - Weak pieces in expendable positions
    """
    score = 0.0
    
    # Calculate position value (back rows more valuable for defense)
    def position_value(r, c, player_id):
        if player_id == 1:
            return 10 - r  # Back rows (6-9) have higher value
        else:
            return r + 1  # Back rows (0-3) have higher value
    
    for piece, (r, c) in placement:
        piece_value = PIECE_RANKS.get(piece, 0)
        pos_value = position_value(r, c, player_id)
        
        # High-value pieces should be in high-value positions
        if piece_value >= 8 and pos_value >= 7:
            score += 0.05
        
        # Low-value pieces can be in lower-value positions
        if piece_value <= 3 and pos_value <= 5:
            score += 0.02
    
    return min(1.0, score)

# Reward: 0.0 to 1.0, scaled to 0-1.0
economy_score = evaluate_piece_economy(placement, player_id)
reward += economy_score * 1.0
```

**Benefit:** Encourages efficient use of board positions

---

### 8. **Adaptive Learning Rewards** (Medium Priority)
**Proposed:** Dynamic rewards based on opponent performance

```python
def calculate_adaptive_reward(placement, player_id, opponent_win_rate, game_length):
    """
    Adjust rewards based on opponent strength and game performance.
    Stronger opponents = higher rewards for good placements.
    """
    base_reward = calculate_setup_agent_reward(...)  # Current reward
    
    # Adjust based on opponent strength
    if opponent_win_rate > 0.6:
        # Strong opponent - reward good placements more
        multiplier = 1.2
    elif opponent_win_rate < 0.4:
        # Weak opponent - less reward (easier to win)
        multiplier = 0.9
    else:
        multiplier = 1.0
    
    # Adjust based on game length (longer games = better setup)
    if game_length > 200:
        multiplier += 0.1
    elif game_length < 50:
        multiplier -= 0.2
    
    return base_reward * multiplier
```

**Benefit:** Adapts to opponent strength, encourages improvement

---

## Implementation Priority

### High Priority (Immediate Impact)
1. **Piece Value Distribution Rewards** - Prevents weak spots
2. **Strategic Piece Positioning** - Core strategic knowledge
3. **Defensive Depth Rewards** - Better defense

### Medium Priority (Significant Improvement)
4. **Piece Synergy Rewards** - Tactical relationships
5. **Vulnerability Assessment Penalties** - Prevents bad placements
6. **Center Control Preparation** - Mid-game strategy
7. **Adaptive Learning Rewards** - Better learning

### Low Priority (Fine-tuning)
8. **Piece Economy Rewards** - Optimization

---

## Additional Setup Agent Improvements

### 1. **Curriculum Learning**
- Start with simple placements, gradually increase complexity
- Early training: focus on flag protection
- Later training: focus on strategic positioning

### 2. **Opponent-Specific Adaptation**
- Learn opponent's common placements
- Adapt setup to counter opponent patterns
- Maintain setup history per opponent

### 3. **Multi-Objective Optimization**
- Balance multiple objectives (protection, aggression, defense)
- Use Pareto-optimal solutions
- Allow different setup styles (aggressive vs defensive)

### 4. **Setup Evaluation Network**
- Train a separate network to evaluate setups
- Use game outcomes to train evaluator
- Faster than playing full games

### 5. **Setup Templates**
- Learn common successful setup patterns
- Use templates as starting points
- Adapt templates based on opponent

---

## Notes
- All rewards should be balanced to avoid overwhelming win/loss rewards
- Test each reward individually before combining
- Monitor setup agent learning curves
- Consider opponent strength when evaluating setups
- Balance between exploration (trying new setups) and exploitation (using good setups)

