import torch
import numpy as np
from distributional_reward import StrategoRewardConfig, UnifiedRewardShaper
from game_state import GameState

def test_reward_hacking_fix():
    config = StrategoRewardConfig.from_training_config()
    # Mocking player_id = 1 (Red, moves up / row indices decrease)
    shaper = UnifiedRewardShaper(player_id=1, config=config)
    
    # Create mock game states
    # Board: 10x10 zero tensor
    board = torch.zeros((10, 10))
    s1 = GameState(board=board.clone(), current_player=1, turn_count=1, game_over=False, 
                  winner=None, move_history=[], uncertainty_mask=None, 
                  revealed_pieces_p1={}, revealed_pieces_p2={})
    
    # 1. Test Territory Advance (Upward move from row 6 to 5)
    # Initial max_row for P1 is 10. Row 5 is new territory.
    action1 = ((6, 0), (5, 0))
    # Fill source and target values for the battle logic inside shaper
    # P1 Piece at (6,0)
    s1.board[6, 0] = 5 # Lieutenant
    s2 = s1.clone()
    s2.board[6, 0] = 0
    s2.board[5, 0] = 5
    
    info = {'num_valid_moves': 10, 'turn_count': 1}
    r1 = shaper(s1, action1, s2, False, None, info)
    
    # Expected: territory_advance (0.05) * positional_weight (0.05) = 0.0025
    # Plus step penalty (-0.0001)
    expected_territory = config.territory_advance * config.positional_weight
    expected_r1 = expected_territory + config.step_penalty
    
    print(f"Move 1 (New Row 5): Reward = {r1:.6f} (Expected ~{expected_r1:.6f})")
    assert abs(r1 - expected_r1) < 1e-7
    
    # 2. Test Anti-Farming (Move from row 5 back to row 6)
    action2 = ((5, 0), (6, 0))
    s3 = s2.clone()
    s3.board[5, 0] = 0
    s3.board[6, 0] = 5
    
    r2 = shaper(s2, action2, s3, False, None, info)
    # Expected: 0 territory reward (row 6 is not < 5), only step penalty
    print(f"Move 2 (Back to Row 6): Reward = {r2:.6f} (Expected {config.step_penalty:.6f})")
    assert abs(r2 - config.step_penalty) < 1e-7
    
    # 3. Test One-time Row Reward (Move back up to row 5 - ALREADY VISITED)
    action3 = ((6, 0), (5, 0))
    s4 = s3.clone()
    s4.board[6, 0] = 0
    s4.board[5, 0] = 5
    r3 = shaper(s3, action3, s4, False, None, info)
    # Expected: 0 territory reward (row 5 is not < current max_row 5), only step penalty
    print(f"Move 3 (Back to Row 5 - Visited): Reward = {r3:.6f} (Expected {config.step_penalty:.6f})")
    assert abs(r3 - config.step_penalty) < 1e-7
    
    # 4. Test New Territory (Move to row 4)
    action4 = ((5, 0), (4, 0))
    s5 = s4.clone()
    s5.board[5, 0] = 0
    s5.board[4, 0] = 5
    r4 = shaper(s4, action4, s5, False, None, info)
    # Expected: territory_advance again
    print(f"Move 4 (New Row 4): Reward = {r4:.6f} (Expected ~{expected_r1:.6f})")
    assert abs(r4 - expected_r1) < 1e-7
    
    # 5. Test Mobility removal (Change 'num_valid_moves' in info)
    info_high = {'num_valid_moves': 100, 'turn_count': 10}
    info_low = {'num_valid_moves': 10, 'turn_count': 11}
    
    # These should give identical step-based rewards if mobility_bonus is gone
    # Using a neutral move (staying in same row if row reward wasn't state-based, but we just check diff)
    action_neutral = ((4, 0), (4, 1))
    s6 = s5.clone()
    s6.board[4, 0] = 0
    s6.board[4, 1] = 5
    
    r_high = shaper(s5, action_neutral, s6, False, None, info_high)
    r_low = shaper(s5, action_neutral, s6, False, None, info_low)
    
    print(f"High Mobility Reward: {r_high:.6f}, Low Mobility Reward: {r_low:.6f}")
    assert abs(r_high - r_low) < 1e-7, "Mobility should not affect rewards anymore!"
    
    print("✅ Reward hacking fix verification passed!")

def test_reward_ordering():
    config = StrategoRewardConfig.from_training_config()
    shaper_p1 = UnifiedRewardShaper(player_id=1, config=config)
    
    board = torch.zeros((10, 10))
    s_base = GameState(board=board, current_player=1, turn_count=1, game_over=False, 
                      winner=None, move_history=[], uncertainty_mask=None, 
                      revealed_pieces_p1={}, revealed_pieces_p2={})
    
    # 1. Immediate Loss (Turn 1)
    # Reward = loss_penalty (-2.0) * outcome_weight (1.0) = -2.0
    r_loss_early = shaper_p1(s_base, None, s_base, True, -1, {'turn_count': 1})
    
    # 2. Late Draw (Turn 2000)
    # Base penalty = draw_penalty (-1.0)
    # Step penalties: 
    #   0-500: 500 * -0.0001 = -0.05
    #   500-1000: 500 * -0.00015 = -0.075
    #   1000-2000: 1000 * -0.0002 = -0.2
    # Total step penalty = -0.325
    # Total reward = -1.0 + -0.325 = -1.325
    
    # We simulate this by manually summing step rewards or just checking the final terminal call
    # The shaper __call__ for terminal state ONLY returns the outcome reward.
    # The step rewards are accumulated during training in the loop.
    # So we need to calculate the TOTAL cumulative reward an agent would get.
    
    def calculate_total_reward(turn_limit, winner):
        total = 0
        shaper = UnifiedRewardShaper(player_id=1, config=config)
        s = s_base.clone()
        for t in range(1, turn_limit + 1):
            done = (t == turn_limit)
            info = {'turn_count': t, 'num_valid_moves': 10}
            # Neutral move (no capture, no territory)
            res = shaper(s, ((4,0), (4,1)) if t%2==0 else ((4,1), (4,0)), s, done, winner if done else None, info)
            total += res
        return total

    total_loss_early = calculate_total_reward(1, -1)
    total_draw_late = calculate_total_reward(2000, 0)
    total_win_late = calculate_total_reward(2000, 1)
    
    print(f"Total Early Loss Reward (T=1): {total_loss_early:.4f}")
    print(f"Total Late Draw Reward (T=2000): {total_draw_late:.4f}")
    print(f"Total Late Win Reward (T=2000): {total_win_late:.4f}")
    
    assert total_win_late > total_draw_late, "Win must be better than Draw!"
    assert total_draw_late > total_loss_early, "Late Draw must be better than Early Loss!"
    
    print("✅ Reward ordering verification passed!")

if __name__ == "__main__":
    test_reward_hacking_fix()
    test_reward_ordering()
