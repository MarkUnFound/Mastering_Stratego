# Reward and Penalty Improvements for Stratego DQN Agents

## Current Reward Structure
- ✅ Move penalties (-0.1 per move)
- ✅ Forward movement rewards (0.5-1.1)
- ✅ Center/territory control (0.2-0.5)
- ✅ Battle outcomes (captures, losses, trades)
- ✅ Tactical rewards (miner defusing bomb, spy capturing marshal, scout reconnaissance)
- ✅ Flag capture (+100.0)
- ✅ Piece preservation
- ✅ Penalties for giving away pieces without exchange
- ✅ Piece advantage reward (count-based)
- ✅ Flag protection reward
- ✅ Defensive positioning reward

## Proposed Additional Rewards and Penalties

### 1. **Piece Value Advantage** (Not Just Count)
**Current:** Only piece count advantage is rewarded
**Proposed:** Reward for maintaining piece value advantage (sum of piece ranks)

```python
# Calculate total piece value for each player
p1_value = sum(abs(piece) for piece in p1_pieces if piece > 0)
p2_value = sum(abs(piece) for piece in p2_pieces if piece < 0)
value_advantage = p1_value - p2_value if current_player == 1 else p2_value - p1_value

if value_advantage > 0:
    # Reward for value advantage (scaled by advantage)
    value_reward = 0.1 * min(value_advantage / 50.0, 1.0) * REWARD_SCALE
    reward += value_reward * phase_multiplier
```

**Benefit:** Encourages preserving high-value pieces, not just quantity

---

### 2. **Piece Coordination / Formation Rewards**
**Proposed:** Reward for pieces advancing together or forming coordinated attacks

```python
# Check if multiple pieces are in enemy territory together
pieces_in_enemy_territory = count_pieces_in_enemy_territory(current_player)
if pieces_in_enemy_territory >= 3:
    coordination_reward = 0.15 * min(pieces_in_enemy_territory / 5.0, 1.0) * REWARD_SCALE
    reward += coordination_reward * phase_multiplier
```

**Benefit:** Encourages coordinated attacks rather than piecemeal advances

---

### 3. **Threat Assessment / Vulnerability Penalties**
**Proposed:** Penalty for leaving pieces vulnerable to attack

```python
# Check if moved piece is now vulnerable (adjacent to enemy pieces)
vulnerable_adjacent_enemies = count_adjacent_enemies(r_to, c_to, current_player)
if vulnerable_adjacent_enemies > 0:
    # Penalty increases with number of threats and piece value
    vulnerability_penalty = 0.1 * vulnerable_adjacent_enemies * (moving_rank / 11.0) * REWARD_SCALE
    reward -= vulnerability_penalty * phase_multiplier
```

**Benefit:** Teaches agents to avoid exposing valuable pieces

---

### 4. **Flag Protection Throughout Game** (Enhanced)
**Current:** Only checks if piece is adjacent to flag
**Proposed:** Reward for maintaining flag protection throughout game

```python
# Calculate flag protection score (number of pieces protecting flag)
flag_protection_count = count_pieces_protecting_flag(current_player)
if flag_protection_count >= 2:
    protection_reward = 0.1 * min(flag_protection_count / 4.0, 1.0) * REWARD_SCALE
    reward += protection_reward * phase_multiplier
elif flag_protection_count == 0:
    # Penalty if flag is unprotected
    protection_penalty = 0.2 * REWARD_SCALE
    reward -= protection_penalty * phase_multiplier
```

**Benefit:** Encourages maintaining flag protection throughout the game

---

### 5. **Endgame Behavior Rewards**
**Proposed:** Different reward structure in endgame (more aggressive, protect flag more)

```python
if game_phase == "end":
    # In endgame, reward aggressive moves toward enemy flag more
    if distance_to_enemy_flag(r_to, c_to, current_player) < distance_to_enemy_flag(r_from, c_from, current_player):
        endgame_aggression_reward = 0.2 * REWARD_SCALE
        reward += endgame_aggression_reward
    
    # Heavier penalty for losing pieces in endgame
    if piece_lost:
        endgame_loss_penalty = 0.3 * (lost_value / 11.0) * REWARD_SCALE
        reward -= endgame_loss_penalty
```

**Benefit:** Encourages appropriate endgame strategy (aggressive flag hunting, careful piece preservation)

---

### 6. **Piece Mobility Rewards**
**Proposed:** Reward for maintaining mobile pieces (not getting stuck)

```python
# Check if piece has multiple valid moves (mobility)
mobility = count_valid_moves_from_position(r_to, c_to, current_player)
if mobility >= 3:
    mobility_reward = 0.05 * (mobility / 8.0) * REWARD_SCALE
    reward += mobility_reward * phase_multiplier
elif mobility == 0:
    # Penalty for getting piece stuck
    stuck_penalty = 0.1 * (moving_rank / 11.0) * REWARD_SCALE
    reward -= stuck_penalty * phase_multiplier
```

**Benefit:** Encourages maintaining piece mobility and avoiding getting trapped

---

### 7. **Control of Key Squares**
**Proposed:** Reward for controlling important squares (near enemy flag, chokepoints)

```python
# Reward for controlling squares near enemy flag
distance_to_enemy_flag = calculate_distance_to_enemy_flag(r_to, c_to, current_player)
if distance_to_enemy_flag <= 2:
    key_square_reward = 0.15 * (1.0 - distance_to_enemy_flag / 2.0) * REWARD_SCALE
    reward += key_square_reward * phase_multiplier

# Reward for controlling center chokepoints (around lakes)
if is_chokepoint(r_to, c_to):
    chokepoint_reward = 0.1 * REWARD_SCALE
    reward += chokepoint_reward * phase_multiplier
```

**Benefit:** Encourages strategic positioning and control of important areas

---

### 8. **Information Gathering Rewards** (Enhanced)
**Current:** Rewards for revealing enemy pieces
**Proposed:** Reward for strategic information gathering

```python
# Reward for revealing enemy pieces with low-value pieces (good trade)
if attacker_rank <= 3 and defender_rank >= 5:
    info_gathering_reward = 0.2 * (defender_rank / 11.0) * REWARD_SCALE
    reward += info_gathering_reward * phase_multiplier

# Reward for revealing multiple enemy pieces in sequence
if consecutive_reveals >= 2:
    consecutive_reveal_bonus = 0.1 * consecutive_reveals * REWARD_SCALE
    reward += consecutive_reveal_bonus * phase_multiplier
```

**Benefit:** Encourages using low-value pieces for reconnaissance

---

### 9. **Tactical Support Rewards**
**Proposed:** Reward for pieces supporting each other tactically

```python
# Reward for pieces that can support each other in battle
supporting_pieces = count_supporting_pieces(r_to, c_to, current_player)
if supporting_pieces >= 2:
    support_reward = 0.1 * min(supporting_pieces / 3.0, 1.0) * REWARD_SCALE
    reward += support_reward * phase_multiplier
```

**Benefit:** Encourages tactical coordination and piece support

---

### 10. **Piece Economy / Value Preservation**
**Proposed:** Reward for maintaining piece value over time

```python
# Track piece value over time
if turn_count % 10 == 0:
    current_piece_value = calculate_total_piece_value(current_player)
    if hasattr(self, '_previous_piece_value'):
        value_change = current_piece_value - self._previous_piece_value[current_player]
        if value_change > 0:
            # Reward for gaining value (captures)
            value_gain_reward = 0.15 * (value_change / 50.0) * REWARD_SCALE
            reward += value_gain_reward
        elif value_change < -5:
            # Penalty for losing significant value
            value_loss_penalty = 0.2 * (abs(value_change) / 50.0) * REWARD_SCALE
            reward -= value_loss_penalty
    self._previous_piece_value[current_player] = current_piece_value
```

**Benefit:** Encourages maintaining piece economy and value over time

---

### 11. **Stalemate Prevention**
**Proposed:** Penalty for moves that lead to stalemate situations

```python
# Check if move reduces available moves significantly
available_moves_before = count_available_moves_before_move(current_player)
available_moves_after = count_available_moves_after_move(current_player)
if available_moves_after < available_moves_before * 0.5:
    stalemate_penalty = 0.15 * REWARD_SCALE
    reward -= stalemate_penalty * phase_multiplier
```

**Benefit:** Prevents agents from getting into stalemate situations


---

### 13. **Bomb Protection Rewards**
**Proposed:** Reward for protecting bombs with pieces

```python
# Check if moved piece is now protecting a bomb
adjacent_bombs = count_adjacent_bombs(r_to, c_to, current_player)
if adjacent_bombs > 0 and moving_rank >= 5:
    bomb_protection_reward = 0.1 * adjacent_bombs * REWARD_SCALE
    reward += bomb_protection_reward * phase_multiplier
```

**Benefit:** Encourages strategic bomb placement and protection

---

### 14. **Scout Mobility Bonus**
**Proposed:** Extra reward for scouts using their long-range mobility

```python
if attacker_type == PieceType.SCOUT:
    distance = abs(row_change) + abs(col_change)
    if distance >= 3:
        scout_mobility_bonus = 0.1 * (distance / 8.0) * REWARD_SCALE
        reward += scout_mobility_bonus * phase_multiplier
```

**Benefit:** Encourages using scouts' unique mobility advantage

---

### 15. **Piece Value Discovery Rewards**
**Proposed:** Reward for discovering enemy high-value pieces through battles

```python
# Extra reward for discovering enemy marshal/general
if defender_rank >= 9 and not was_previously_revealed:
    discovery_reward = 0.3 * (defender_rank / 11.0) * REWARD_SCALE
    reward += discovery_reward * phase_multiplier
```

**Benefit:** Encourages strategic information gathering about enemy high-value pieces

---

## Implementation Priority

### High Priority (Immediate Impact)
1. **Piece Value Advantage** - Better than just count
2. **Threat Assessment / Vulnerability Penalties** - Prevents bad moves
3. **Endgame Behavior Rewards** - Critical for endgame play
4. **Flag Protection Throughout Game** - Essential defensive behavior

### Medium Priority (Significant Improvement)
5. **Piece Coordination / Formation Rewards** - Improves tactical play
6. **Control of Key Squares** - Strategic positioning
7. **Information Gathering Rewards** - Better reconnaissance
8. **Piece Mobility Rewards** - Prevents getting stuck

### Low Priority (Fine-tuning)
9. **Tactical Support Rewards** - Advanced coordination
10. **Piece Economy / Value Preservation** - Long-term strategy
11. **Stalemate Prevention** - Edge case handling
13. **Bomb Protection Rewards** - Defensive strategy
14. **Scout Mobility Bonus** - Piece-specific optimization
15. **Piece Value Discovery Rewards** - Information value

## Notes
- All rewards should be scaled by `REWARD_SCALE` (10.0) for consistency
- Phase multipliers should be applied appropriately
- Rewards should be balanced to avoid overwhelming the flag capture reward (+100.0)
- Consider testing each reward individually before combining them

