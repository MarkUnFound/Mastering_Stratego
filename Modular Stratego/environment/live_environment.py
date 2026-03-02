
class LiveStrategoEnvironment:
    """Enhanced Stratego environment with live visualization and bug fixes."""
    
    def __init__(self, device, show_live_view=True, show_agent_views=False):
        self.device = device
        self.board = Board(device)
        self.battle_resolver = BattleResolver()
        self.directions = torch.tensor([(0, 1), (0, -1), (1, 0), (-1, 0)], device=device)
        
        # Visualization components
        self.show_live_view = show_live_view
        self.show_agent_views = show_agent_views
        self.live_viewer = LiveGameViewer() if show_live_view else None
        self.agent_viewer = RestrictedGameViewer() if show_agent_views else None
        
        # Game state
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        
        # Bug fix: Track piece ownership more carefully
        self.piece_ownership = {}  # Maps (r, c) -> player_id
        
        self.reset()
        
    def reset(self) -> GameState:
        """Reset the environment to start a new game."""
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        self.piece_ownership = {}
        
        # Setup pieces in starting positions
        p1_pieces = self._generate_pieces()
        p2_pieces = self._generate_pieces()
        
        # Verify we have exactly one flag per player
        p1_flag_count = p1_pieces.count(PieceType.FLAG)
        p2_flag_count = p2_pieces.count(PieceType.FLAG)
        
        if p1_flag_count != 1:
            print(f"Warning: Player 1 has {p1_flag_count} flags, expected 1")
        if p2_flag_count != 1:
            print(f"Warning: Player 2 has {p2_flag_count} flags, expected 1")
        
        # Place pieces on the board
        p1_positions = self._get_p1_positions()
        p2_positions = self._get_p2_positions()
        
        # Ensure we have exactly 40 positions for each player
        p1_positions = p1_positions[:40]
        p2_positions = p2_positions[:40]
        
        # Ensure we have exactly 40 pieces for each player
        p1_pieces = p1_pieces[:40]
        p2_pieces = p2_pieces[:40]
        
        self.board.setup_pieces(
            [(piece, (r, c)) for piece, (r, c) in zip(p1_pieces, p1_positions)],
            [(piece, (r, c)) for piece, (r, c) in zip(p2_pieces, p2_positions)]
        )
        
        # Bug fix: Initialize piece ownership tracking
        for piece, (r, c) in zip(p1_pieces, p1_positions):
            self.piece_ownership[(r, c)] = 1
        for piece, (r, c) in zip(p2_pieces, p2_positions):
            self.piece_ownership[(r, c)] = 2
            
        # Track flag positions
        for r, c in p1_positions:
            if self.board.actual_board[r, c].item() == PieceType.FLAG.value:
                self.p1_flag_position = (r, c)
        for r, c in p2_positions:
            if self.board.actual_board[r, c].item() == -PieceType.FLAG.value:
                self.p2_flag_position = (r, c)
                
        # Initialize visualization
        if self.live_viewer:
            self.live_viewer.update_display(
                self.board.actual_board, self.current_player, self.turn_count
            )
            self.live_viewer.show()
            
        if self.agent_viewer:
            self.agent_viewer.update_display(
                self.board.visible_board_p1, self.board.visible_board_p2,
                self.current_player, self.turn_count
            )
            self.agent_viewer.show()
        
        return self._get_game_state()
        
    def _generate_pieces(self) -> List[PieceType]:
        """Generate a list of pieces for one player."""
        # Standard Stratego setup: 1 flag, 1 spy, 6 bombs, 1 marshal, 1 general, etc.
        pieces = [PieceType.FLAG, PieceType.SPY] + [PieceType.BOMB]*6 + [PieceType.MARSHAL] + \
                 [PieceType.GENERAL] + [PieceType.COLONEL]*2 + [PieceType.MAJOR]*3 + \
                 [PieceType.CAPTAIN]*4 + [PieceType.LIEUTENANT]*4 + [PieceType.SERGEANT]*4 + \
                 [PieceType.MINER]*5 + [PieceType.SCOUT]*8
        
        # Ensure exactly one flag
        flag_count = pieces.count(PieceType.FLAG)
        if flag_count > 1:
            # Remove extra flags
            while pieces.count(PieceType.FLAG) > 1:
                pieces.remove(PieceType.FLAG)
        elif flag_count == 0:
            # Add missing flag
            pieces.append(PieceType.FLAG)
            
        # Ensure exactly 40 pieces total (standard Stratego)
        if len(pieces) > 40:
            # Remove extra pieces (shouldn't happen with correct setup)
            while len(pieces) > 40 and pieces:
                if pieces[-1] != PieceType.FLAG:  # Don't remove the flag
                    pieces.pop()
                else:
                    break
        elif len(pieces) < 40:
            # Add scouts to fill (standard practice)
            while len(pieces) < 40:
                pieces.append(PieceType.SCOUT)
                
        return pieces
        
    def _get_p1_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 1."""
        positions = [(r, c) for r in range(6, 10) for c in range(10)]
        # Remove lake positions
        lake_positions = set((r.item(), c.item()) for r, c in self.board.lakes)
        positions = [pos for pos in positions if pos not in lake_positions]
        random.shuffle(positions)
        return positions[:40]
        
    def _get_p2_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 2."""
        positions = [(r, c) for r in range(0, 4) for c in range(10)]
        # Remove lake positions
        lake_positions = set((r.item(), c.item()) for r, c in self.board.lakes)
        positions = [pos for pos in positions if pos not in lake_positions]
        random.shuffle(positions)
        return positions[:40]
        
    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all valid moves for the current player with enhanced bug fixes."""
        moves = []
        visible_board = self.board.get_visible_board(self.current_player)
        player_pieces = torch.nonzero((visible_board * self.current_player > 0) & (visible_board != LAKE_SQUARE))
        
        for r_from, c_from in player_pieces:
            r, c = r_from.item(), c_from.item()
            piece_value = visible_board[r, c].item()
            piece_type = PieceType(abs(piece_value))
            
            # Bug fix: Enhanced check - Flags and bombs cannot move
            # Also verify ownership to prevent random switching
            if (piece_type in [PieceType.FLAG, PieceType.BOMB] or 
                self.piece_ownership.get((r, c), 0) != (1 if self.current_player == 1 else 2)):
                continue
                
            # Scout can move any distance in a straight line
            if piece_type == PieceType.SCOUT:
                for dr, dc in self.directions:
                    for i in range(1, BOARD_SIZE):
                        r_to, c_to = r + i * dr.item(), c + i * dc.item()
                        # Check if target is valid
                        if not self.board.is_valid_target(self.current_player, r_to, c_to):
                            break
                        # Check if square is occupied by any piece (including hidden pieces)
                        target_value = visible_board[r_to, c_to].item()
                        if target_value != EMPTY_SQUARE:
                            # Can capture enemy piece but cannot move through any piece
                            moves.append(((r, c), (r_to, c_to)))
                            break
                        moves.append(((r, c), (r_to, c_to)))
            else:
                # Other pieces move one square
                for dr, dc in self.directions:
                    r_to, c_to = r + dr.item(), c + dc.item()
                    if self.board.is_valid_target(self.current_player, r_to, c_to):
                        moves.append(((r, c), (r_to, c_to)))
                        
        return moves
        
    def step(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[GameState, float, bool, Dict]:
        """Execute a move and return the new state with enhanced visualization."""
        if self.game_over:
            return self._get_game_state(), 0.0, True, {"winner": self.winner}
            
        (r_from, c_from), (r_to, c_to) = action
        reward = -0.01  # Small penalty for each move
        
        # Validate move ownership (bug fix)
        if self.piece_ownership.get((r_from, c_from), 0) != (1 if self.current_player == 1 else 2):
            # Invalid move - piece doesn't belong to current player
            return self._get_game_state(), -1.0, False, {"error": "Invalid piece ownership"}
        
        # Get pieces involved in the move
        moving_piece_value = self.board.actual_board[r_from, c_from].item()
        target_piece_value = self.board.actual_board[r_to, c_to].item()
        
        # Handle battle or simple move
        if target_piece_value != EMPTY_SQUARE and target_piece_value != LAKE_SQUARE:
            # Battle occurs
            attacker_type = PieceType(abs(moving_piece_value))
            defender_type = PieceType(abs(target_piece_value))
            
            # Reveal pieces to both players
            self.board.reveal_pieces((r_from, c_from), (r_to, c_to))
            self.revealed_pieces_p1[(r_from, c_from)] = abs(moving_piece_value)
            self.revealed_pieces_p2[(r_from, c_from)] = abs(moving_piece_value)
            self.revealed_pieces_p1[(r_to, c_to)] = abs(target_piece_value)
            self.revealed_pieces_p2[(r_to, c_to)] = abs(target_piece_value)
            
            # Determine player ownership for battle resolution using piece_ownership tracking
            attacker_player = self.piece_ownership.get((r_from, c_from), 0)
            defender_player = self.piece_ownership.get((r_to, c_to), 0)
            result = self.battle_resolver.resolve_battle(attacker_type, defender_type, attacker_player, defender_player)
            
            if result == 1:  # Attacker wins
                # Bug fix: Update ownership tracking
                del self.piece_ownership[(r_to, c_to)]  # Remove defender
                self.piece_ownership[(r_to, c_to)] = self.piece_ownership[(r_from, c_from)]  # Move attacker
                del self.piece_ownership[(r_from, c_from)]
                
                self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
                reward += 0.1 * abs(target_piece_value)
                if defender_type == PieceType.FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    reward += 1.0
                    # Update flag tracking
                    self.p2_flag_position = None if self.current_player == 1 else self.p2_flag_position
                    self.p1_flag_position = None if self.current_player == 2 else self.p1_flag_position
            elif result == -1:  # Defender wins
                # Remove attacker
                self.board.actual_board[r_from, c_from] = EMPTY_SQUARE
                if self.current_player == 1:
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                else:
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                
                # Bug fix: Update ownership tracking
                del self.piece_ownership[(r_from, c_from)]  # Remove attacker
                
                # Check if attacker was a flag
                attacker_type = PieceType(abs(moving_piece_value))
                if attacker_type == PieceType.FLAG:
                    # Update flag tracking
                    if self.current_player == 1:
                        self.p1_flag_position = None
                    else:
                        self.p2_flag_position = None
                reward -= 0.1 * abs(moving_piece_value)
            else:  # Tie
                # Both pieces are removed
                self.board.actual_board[r_from, c_from] = EMPTY_SQUARE
                self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                self.board.actual_board[r_to, c_to] = EMPTY_SQUARE
                self.board.visible_board_p1[r_to, c_to] = EMPTY_SQUARE
                self.board.visible_board_p2[r_to, c_to] = EMPTY_SQUARE
                # Bug fix: Update ownership tracking
                del self.piece_ownership[(r_from, c_from)]
                del self.piece_ownership[(r_to, c_to)]
                
                # Check if either piece was a flag
                attacker_type = PieceType(abs(moving_piece_value))
                defender_type = PieceType(abs(target_piece_value))
                if attacker_type == PieceType.FLAG:
                    # Update flag tracking
                    if self.current_player == 1:
                        self.p1_flag_position = None
                    else:
                        self.p2_flag_position = None
                if defender_type == PieceType.FLAG:
                    # Update flag tracking
                    # Current player is capturing the flag, so the flag belongs to the opposite player
                    if self.current_player == 1:  # Player 1 is capturing Player 2's flag
                        self.p2_flag_position = None
                    else:  # Player 2 is capturing Player 1's flag
                        self.p1_flag_position = None
        else:
            # Simple move to empty square
            # Bug fix: Update ownership tracking
            del self.piece_ownership[(r_from, c_from)]
            self.piece_ownership[(r_to, c_to)] = 1 if self.current_player == 1 else 2
            
            # Check if moving piece is a flag
            moving_piece_type = PieceType(abs(moving_piece_value))
            if moving_piece_type == PieceType.FLAG:
                # Update flag tracking
                if self.current_player == 1:
                    self.p1_flag_position = (r_to, c_to)
                else:
                    self.p2_flag_position = (r_to, c_to)
            
            self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
            
        self.turn_count += 1
        self.move_history.append(action)
        
        # Update visualizations
        if self.live_viewer:
            self.live_viewer.update_display(
                self.board.actual_board, -self.current_player, self.turn_count, 
                action, self.game_over, self.winner,
                p1_flag_pos=self.p1_flag_position, p2_flag_pos=self.p2_flag_position
            )
            
        if self.agent_viewer:
            self.agent_viewer.update_display(
                self.board.visible_board_p1, self.board.visible_board_p2,
                -self.current_player, self.turn_count
            )
        
        self.current_player *= -1
        
        # Check for game end conditions
        self._check_game_end()
        
        return self._get_game_state(), reward, self.game_over, {"winner": self.winner}
        
    def _check_game_end(self):
        """Checks for game-ending conditions using tracked flag positions."""
        # Check if flags still exist
        p1_flag_exists = self.p1_flag_position is not None
        p2_flag_exists = self.p2_flag_position is not None
        
        # Verify flag positions are still valid
        if p1_flag_exists:
            r, c = self.p1_flag_position
            if self.board.actual_board[r, c].item() != PieceType.FLAG.value:
                p1_flag_exists = False
                self.p1_flag_position = None
        
        if p2_flag_exists:
            r, c = self.p2_flag_position
            if self.board.actual_board[r, c].item() != -PieceType.FLAG.value:
                p2_flag_exists = False
                self.p2_flag_position = None
        
        if not p1_flag_exists and not p2_flag_exists:
            # Both flags captured - this shouldn't happen in normal play
            self.game_over = True
            self.winner = 0  # Draw
        elif not p1_flag_exists:
            # Player 1's flag captured
            self.game_over = True
            self.winner = -1  # Player 2 wins
        elif not p2_flag_exists:
            # Player 2's flag captured
            self.game_over = True
            self.winner = 1   # Player 1 wins
        
        # Check if current player has any valid moves
        if not self.game_over and not self.get_valid_moves():
            self.game_over = True
            self.winner = -self.current_player # Player who cannot move loses
        
        # Smart draw detection
        if not self.game_over and self.turn_count > 300:  # Start checking after 300 moves
            # Check for repetitive positions (same piece arrangement)
            if self._is_position_repetitive():
                self.game_over = True
                self.winner = 0
                return
            # Check for minimal piece movement (stalemate)
            if self._is_stalemate():
                self.game_over = True
                self.winner = 0
                
    def _is_position_repetitive(self) -> bool:
        """Check if the game is in a repetitive state."""
        if len(self.move_history) < 8:
            return False
            
        # Check if the last 4 moves repeat the previous 4 moves
        recent_moves = self.move_history[-8:]
        return recent_moves[:4] == recent_moves[4:]
        
    def _is_stalemate(self) -> bool:
        """Check if the game is in a stalemate."""
        # Check if the last 10 moves have any piece movement
        recent_moves = self.move_history[-20:]
        for move in recent_moves:
            r_from, c_from = move[0]
            r_to, c_to = move[1]
            if r_from != r_to or c_from != c_to:
                return False
        return True
        
    def _get_game_state(self) -> GameState:
        """Get the current game state."""
        # Create uncertainty mask (1 where pieces are hidden, 0 where visible)
        visible_board = self.board.get_visible_board(self.current_player)
        uncertainty_mask = (visible_board == -1).float()  # HIDDEN_PIECE = -1
        
        return GameState(
            board=visible_board,
            current_player=self.current_player,
            turn_count=self.turn_count,
            game_over=self.game_over,
            winner=self.winner,
            move_history=self.move_history.copy(),
            uncertainty_mask=uncertainty_mask,
            revealed_pieces_p1=self.revealed_pieces_p1.copy(),
            revealed_pieces_p2=self.revealed_pieces_p2.copy()
        )
        
    def close_viewers(self):
        """Close all visualization windows."""
        if self.live_viewer:
            self.live_viewer.close()
        if self.agent_viewer:
            self.agent_viewer.close()
            
    def pause_for_viewing(self, seconds: float = 1.0):
        """Pause execution to allow viewing of the current state."""
        time.sleep(seconds)
