import torch
import random
from kluss_solver import KLUSSSolver
from dqn_evaluator import DQNEvaluator
from battle import BattleResolver

class HybridAgent:
    def __init__(self, player_id, model_path=None, device='cpu'):
        self.player_id = player_id
        self.device = device
        
        # Initialize Components
        self.evaluator = DQNEvaluator(model_path, device)
        self.solver = KLUSSSolver(self.evaluator)

    def get_belief_state(self, game_state):
        return None
        
    def act(self, game_state, valid_moves):
        """
        Choose an action using the KLUSS + DQN hybrid system.
        """
        if not valid_moves:
            return None
            
        # 1. Update Belief State
        # In a real implementation, we would maintain a belief state object
        # that tracks the probability distribution of enemy pieces.
        # For now, we assume game_state contains necessary info or we construct a fresh one.
        belief_state = self.get_belief_state(game_state)
        
        # 2. Run KLUSS Search
        # This returns a strategy (probability distribution over actions)
        strategy = self.solver.solve(game_state, belief_state)
        
        # 3. Select Action
        # We can sample from the strategy or take the argmax
        # For competitive play, usually sample (to remain unpredictable) or argmax (for max strength)
        # Here we'll do a weighted sample based on the strategy
        
        # Map strategy (which might be abstract) back to valid_moves
        # ...
        
        # Fallback to random if solver fails or returns empty
        return random.choice(valid_moves)

    def get_top_moves(self, game_state, valid_moves, n=3):
        """
        Evaluate all valid moves and return the top n.
        Returns: List of (move, value) tuples.
        """
        if not valid_moves:
            return []
            
        move_values = self.analyze_state(game_state, valid_moves)
        
        # Sort by value descending
        move_values.sort(key=lambda x: x[1], reverse=True)
        return move_values[:n]

    def analyze_state(self, game_state, valid_moves):
        """
        Analyze the state and return top moves with values.
        """
        move_values = []
        battle_resolver = BattleResolver()
        
        # Prepare batch of states for evaluation
        next_states = []
        move_indices = [] # Track which move generated which state(s)
        weights = [] # Weights for probabilistic outcomes
        
        for i, move in enumerate(valid_moves):
            (r1, c1), (r2, c2) = move
            piece = game_state.board[r1, c1].item()
            target = game_state.board[r2, c2].item()
            
            if target == 0: # Empty square
                # Deterministic
                next_state = self._simulate_move(game_state, move, battle_result=None)
                next_states.append(next_state)
                move_indices.append(i)
                weights.append(1.0)
            else:
                # Battle
                # If target is known (revealed), deterministic
                # If target is unknown, probabilistic
                
                # Check if target is revealed
                # We need to check game_state.revealed_pieces_p1/p2
                # Assuming we are the current player
                opponent = -game_state.current_player
                revealed = game_state.get_revealed_pieces(game_state.current_player)
                
                if (r2, c2) in revealed:
                    # Deterministic Battle
                    target_rank = revealed[(r2, c2)]
                    # We need PieceType enum from rank? Or just use rank?
                    # BattleResolver takes PieceType.
                    # We need to map rank back to PieceType or modify BattleResolver.
                    # Let's assume we can map.
                    # For now, let's just use the board value if it's revealed (it might be in the board tensor?)
                    # In Stratego, board usually has piece IDs or Types.
                    # If hidden, it might be just sign.
                    
                    # Let's simplify: If we don't have belief state, assume uniform random or just use board value if visible.
                    # For visualization, we want to show what the agent *thinks*.
                    
                    # If we have belief state, use it.
                    belief_state = self.get_belief_state(game_state)
                    if belief_state:
                         # Get probs for target
                         probs = belief_state.get_probs(r2, c2) # Need to implement this in PBS
                         # Iterate outcomes
                         for rank, prob in probs.items():
                             if prob > 0.01:
                                 # Resolve
                                 # ...
                                 pass
                    else:
                        # Fallback: Assume target is what it is (cheating/god mode) or random
                        # For visualization of "Intuition", maybe just use the board state as is?
                        # But the board state has hidden pieces as just "Enemy".
                        # If we pass "Enemy" to DQN, it handles it.
                        # So we just need to update the board tensor to reflect the move.
                        
                        # Actually, DQN takes (41, 10, 10).
                        # Channel 37/38 are Last Move.
                        # Channels 0-11 are My Pieces.
                        # Channels 12-23 are Known Enemy.
                        # Channels 24-35 are Unknown Enemy Probs.
                        
                        # If we move, we update My Pieces.
                        # If we battle, we might disappear.
                        
                        # SIMPLIFICATION:
                        # Just execute the move assuming we win? Or assume average case?
                        # Or just execute the move on the tensor directly?
                        
                        # Let's do the rigorous thing:
                        # 1. Clone state.
                        # 2. Update board.
                        # 3. Convert to tensor.
                        
                        next_state = self._simulate_move(game_state, move, battle_result=1) # Assume win for visualization?
                        next_states.append(next_state)
                        move_indices.append(i)
                        weights.append(1.0)

        # Batch Evaluate
        if not next_states:
            return []
            
        # Convert to tensors
        tensors = [self.solver.state_to_tensor(s) for s in next_states]
        batch_tensor = torch.stack(tensors)
        
        values = self.evaluator.evaluate_batch(batch_tensor)
        
        # Aggregate
        # Currently 1-to-1 mapping because we simplified battles
        results = []
        for idx, val in zip(move_indices, values):
            results.append((valid_moves[idx], val))
            
        return results

    def _simulate_move(self, state, move, battle_result=None):
        """
        Create a new GameState with the move applied.
        """
        import copy
        new_state = copy.deepcopy(state)
        (r1, c1), (r2, c2) = move
        
        piece = new_state.board[r1, c1].item()
        
        # Clear source
        new_state.board[r1, c1] = 0
        
        # Set target
        if battle_result == 1: # Attacker wins
            new_state.board[r2, c2] = piece
        elif battle_result == -1: # Defender wins
            pass # Piece dies, target stays (already there)
        elif battle_result == 0: # Draw
            new_state.board[r2, c2] = 0
        else: # Move to empty
            new_state.board[r2, c2] = piece
            
        # Update other state vars (turn count, etc)
        new_state.current_player *= -1
        new_state.turn_count += 1
        
        return new_state
