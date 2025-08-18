import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from stratego_env import StrategoEnv, Piece
from dqn_agent import DQNAgent
import threading
import time
import os

class StrategoGUI:
    """GUI interface for playing Stratego against trained agents"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stratego DQN - Human vs AI")
        self.root.geometry("1000x800")
        
        # Game state
        self.env = StrategoEnv()
        self.agent = None
        self.human_player = 0  # Human is red (0), AI is blue (1)
        self.game_active = False
        self.selected_piece = None
        self.valid_moves = []
        
        # Colors and styling
        self.colors = {
            'red': '#FF6B6B',
            'blue': '#4ECDC4',
            'empty': '#F7F7F7',
            'water': '#74B9FF',
            'selected': '#FFD93D',
            'valid_move': '#6BCF7F',
            'border': '#2D3436'
        }
        
        # Piece symbols for display
        self.piece_symbols = {
            0: '',   # Empty
            -1: '≈',  # Water
            # Red pieces (Human)
            1: '🏴', 2: '🕵', 3: '👤', 4: '⛏️', 5: '🎖️', 6: '👨‍✈️',
            7: '👮', 8: '🎖️', 9: '🎖️', 10: '⭐', 11: '👑', 12: '💣',
            # Blue pieces (AI)
            13: '🏁', 14: '🕵', 15: '👥', 16: '⛏️', 17: '🎖️', 18: '👨‍✈️',
            19: '👮', 20: '🎖️', 21: '🎖️', 22: '⭐', 23: '👑', 24: '💥'
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Game board
        self.setup_game_board(main_frame)
        
        # Info panel
        self.setup_info_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup control buttons and options"""
        control_frame = ttk.LabelFrame(parent, text="Game Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model loading
        ttk.Label(control_frame, text="Load AI Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_path_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.model_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Load", command=self.load_model).grid(row=0, column=3, padx=5)
        
        # Game controls
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="New Game", command=self.new_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Hint", command=self.get_hint).pack(side=tk.LEFT, padx=5)
        
        # Difficulty selection
        ttk.Label(control_frame, text="AI Difficulty:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.difficulty_var = tk.StringVar(value="Normal")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var, 
                                       values=["Easy", "Normal", "Hard"], state="readonly")
        difficulty_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        
    def setup_game_board(self, parent):
        """Setup the game board"""
        board_frame = ttk.LabelFrame(parent, text="Game Board", padding=10)
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create board grid
        self.board_buttons = []
        for r in range(10):
            row = []
            for c in range(10):
                btn = tk.Button(
                    board_frame,
                    width=4,
                    height=2,
                    font=('Arial', 12, 'bold'),
                    command=lambda row=r, col=c: self.on_square_click(row, col)
                )
                btn.grid(row=r, column=c, padx=1, pady=1)
                row.append(btn)
            self.board_buttons.append(row)
        
        # Add coordinate labels
        for i in range(10):
            tk.Label(board_frame, text=str(i), font=('Arial', 8)).grid(row=i, column=10, padx=5)
            tk.Label(board_frame, text=str(i), font=('Arial', 8)).grid(row=10, column=i, pady=5)
        
    def setup_info_panel(self, parent):
        """Setup information panel"""
        info_frame = ttk.LabelFrame(parent, text="Game Information", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Game status
        self.status_var = tk.StringVar(value="Load a model to start playing")
        ttk.Label(info_frame, textvariable=self.status_var, wraplength=200).pack(pady=5)
        
        # Current player
        self.current_player_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.current_player_var, 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Move history
        ttk.Label(info_frame, text="Move History:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        self.history_text = tk.Text(info_frame, width=30, height=15, font=('Arial', 9))
        history_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scroll.set)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Captured pieces
        ttk.Label(info_frame, text="Captured:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        self.captured_var = tk.StringVar(value="Red: 0, Blue: 0")
        ttk.Label(info_frame, textvariable=self.captured_var).pack(pady=5)
        
        # Game statistics
        ttk.Label(info_frame, text="Statistics:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        self.stats_var = tk.StringVar(value="Moves: 0\nTime: 0:00")
        ttk.Label(info_frame, textvariable=self.stats_var).pack(pady=5)
        
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select DQN Model",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found")
            return
        
        try:
            # Initialize agent
            state_size = self.env.get_state_space_size()
            action_size = self.env.get_action_space_size()
            
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                player_id=1,  # AI is player 1 (blue)
                epsilon=0.0   # No exploration for playing
            )
            
            # Load model
            if self.agent.load_model(model_path):
                self.agent.set_training_mode(False)
                self.status_var.set("Model loaded successfully! Click 'New Game' to start.")
                messagebox.showinfo("Success", "Model loaded successfully!")
            else:
                self.status_var.set("Failed to load model")
                messagebox.showerror("Error", "Failed to load model")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
    def new_game(self):
        """Start a new game"""
        if self.agent is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        # Reset environment
        self.env.reset()
        self.game_active = True
        self.selected_piece = None
        self.valid_moves = []
        
        # Clear history
        self.history_text.delete(1.0, tk.END)
        
        # Update display
        self.update_board_display()
        self.update_status()
        
        # Start game timer
        self.game_start_time = time.time()
        self.update_timer()
        
    def reset_game(self):
        """Reset current game"""
        self.game_active = False
        self.selected_piece = None
        self.valid_moves = []
        self.env.reset()
        self.update_board_display()
        self.status_var.set("Game reset. Click 'New Game' to start.")
    
    def on_square_click(self, row, col):
        """Handle square click"""
        if not self.game_active or self.env.game_over:
            return
        
        # Only allow human moves on human turn
        if self.env.current_player != self.human_player:
            return
        
        if self.selected_piece is None:
            # Select piece
            piece = self.env.board[row, col]
            if self.env._is_player_piece(piece, self.human_player) and self.env._can_piece_move(piece):
                self.selected_piece = (row, col)
                self.valid_moves = self.get_valid_moves_for_piece(row, col)
                self.update_board_display()
        else:
            # Try to move piece
            from_r, from_c = self.selected_piece
            to_r, to_c = row, col
            
            # Check if this is a valid move
            action = self.env._encode_action(from_r, from_c, to_r, to_c)
            valid_actions = self.env.get_valid_actions(self.human_player)
            
            if action in valid_actions:
                # Execute move
                state = self.env.get_state()
                next_state, reward, done, info = self.env.step(action)
                
                # Log move
                self.log_move(from_r, from_c, to_r, to_c, "Human")
                
                # Clear selection
                self.selected_piece = None
                self.valid_moves = []
                
                # Update display
                self.update_board_display()
                
                if done:
                    self.end_game(info)
                else:
                    # AI turn
                    self.current_player_var.set("AI is thinking...")
                    self.root.update()
                    self.root.after(1000, self.ai_move)  # Delay for better UX
            else:
                # Invalid move, clear selection
                self.selected_piece = None
                self.valid_moves = []
                self.update_board_display()
    
    def get_valid_moves_for_piece(self, row, col):
        """Get valid moves for a specific piece"""
        valid_moves = []
        piece = self.env.board[row, col]
        
        # Check all possible moves
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            
            # Scout can move multiple squares
            if self.env._is_scout(piece):
                distance = 1
                while self.env._is_valid_move(row, col, nr, nc, self.human_player):
                    valid_moves.append((nr, nc))
                    
                    # If we hit an enemy piece, can't go further
                    if self.env.board[nr, nc] != 0:
                        break
                    
                    # Continue in same direction
                    distance += 1
                    nr, nc = row + dr * distance, col + dc * distance
            else:
                if self.env._is_valid_move(row, col, nr, nc, self.human_player):
                    valid_moves.append((nr, nc))
        
        return valid_moves
    
    def ai_move(self):
        """Execute AI move"""
        if not self.game_active or self.env.game_over:
            return
        
        if self.env.current_player != 1:  # AI is player 1
            return
        
        try:
            # Get AI action
            state = self.env.get_state()
            valid_actions = self.env.get_valid_actions(1)
            
            if not valid_actions:
                self.end_game({"winner": 0, "no_moves": True})
                return
            
            # Adjust AI difficulty
            if self.difficulty_var.get() == "Easy":
                # Increase randomness
                import random
                if random.random() < 0.3:
                    action = random.choice(valid_actions)
                else:
                    action = self.agent.select_action(state, valid_actions, training=False)
            elif self.difficulty_var.get() == "Hard":
                # Use best action
                action = self.agent.select_action(state, valid_actions, training=False)
            else:  # Normal
                # Small amount of randomness
                import random
                if random.random() < 0.1:
                    action = random.choice(valid_actions)
                else:
                    action = self.agent.select_action(state, valid_actions, training=False)
            
            # Execute move
            from_r, from_c, to_r, to_c = self.env._decode_action(action)
            next_state, reward, done, info = self.env.step(action)
            
            # Log move
            self.log_move(from_r, from_c, to_r, to_c, "AI")
            
            # Update display
            self.update_board_display()
            
            if done:
                self.end_game(info)
            else:
                self.update_status()
                
        except Exception as e:
            messagebox.showerror("Error", f"AI move error: {str(e)}")
            self.game_active = False
    
    def update_board_display(self):
        """Update the visual board display"""
        for r in range(10):
            for c in range(10):
                btn = self.board_buttons[r][c]
                piece = self.env.board[r, c]
                
                # Set button text
                if piece == -1:  # Water
                    btn.config(text=self.piece_symbols[piece], bg=self.colors['water'])
                elif piece == 0:  # Empty
                    btn.config(text='', bg=self.colors['empty'])
                elif self.env._is_player_piece(piece, 0):  # Red (Human)
                    if self.human_player == 0:
                        # Show human pieces
                        btn.config(text=self.get_piece_display(piece), bg=self.colors['red'])
                    else:
                        # Hide opponent pieces
                        btn.config(text='?', bg=self.colors['red'])
                else:  # Blue (AI)
                    if self.human_player == 1:
                        # Show human pieces (if human is blue)
                        btn.config(text=self.get_piece_display(piece), bg=self.colors['blue'])
                    else:
                        # Hide AI pieces
                        btn.config(text='?', bg=self.colors['blue'])
                
                # Highlight selected piece
                if self.selected_piece == (r, c):
                    btn.config(bg=self.colors['selected'])
                
                # Highlight valid moves
                elif (r, c) in self.valid_moves:
                    current_bg = btn.cget('bg')
                    if current_bg == self.colors['empty']:
                        btn.config(bg=self.colors['valid_move'])
                    else:
                        btn.config(relief=tk.RAISED, bd=3)
                else:
                    btn.config(relief=tk.FLAT, bd=1)
    
    def get_piece_display(self, piece):
        """Get display representation of piece"""
        piece_names = {
            # Red pieces
            1: 'F', 2: 'S', 3: '2', 4: '3', 5: '4', 6: '5',
            7: '6', 8: '7', 9: '8', 10: '9', 11: '10', 12: 'B',
            # Blue pieces  
            13: 'F', 14: 'S', 15: '2', 16: '3', 17: '4', 18: '5',
            19: '6', 20: '7', 21: '8', 22: '9', 23: '10', 24: 'B'
        }
        return piece_names.get(piece, '?')
    
    def update_status(self):
        """Update game status"""
        if self.env.game_over:
            winner = "Draw" if self.env.winner is None else f"{'Human' if self.env.winner == self.human_player else 'AI'} wins!"
            self.status_var.set(f"Game Over - {winner}")
            self.current_player_var.set("")
        else:
            current = "Your turn" if self.env.current_player == self.human_player else "AI turn"
            self.current_player_var.set(current)
            self.status_var.set("Game in progress")
    
    def log_move(self, from_r, from_c, to_r, to_c, player):
        """Log a move to history"""
        piece = self.env.board[to_r, to_c] if self.env.board[to_r, to_c] != 0 else "moved"
        move_text = f"{player}: ({from_r},{from_c}) → ({to_r},{to_c})\n"
        
        self.history_text.insert(tk.END, move_text)
        self.history_text.see(tk.END)
    
    def update_timer(self):
        """Update game timer"""
        if self.game_active and hasattr(self, 'game_start_time'):
            elapsed = int(time.time() - self.game_start_time)
            minutes, seconds = divmod(elapsed, 60)
            time_str = f"{minutes}:{seconds:02d}"
            
            move_count = self.env.move_count
            self.stats_var.set(f"Moves: {move_count}\nTime: {time_str}")
            
            # Schedule next update
            self.root.after(1000, self.update_timer)
    
    def end_game(self, info):
        """End the current game"""
        self.game_active = False
        
        winner_text = ""
        if info.get('winner') is not None:
            winner = info['winner']
            winner_text = "You win!" if winner == self.human_player else "AI wins!"
        else:
            winner_text = "It's a draw!"
        
        self.status_var.set(f"Game Over - {winner_text}")
        self.current_player_var.set("")
        
        # Show game over dialog
        messagebox.showinfo("Game Over", winner_text)
    
    def get_hint(self):
        """Get AI hint for current position"""
        if not self.game_active or self.agent is None:
            return
        
        if self.env.current_player != self.human_player:
            messagebox.showinfo("Hint", "It's not your turn!")
            return
        
        try:
            state = self.env.get_state()
            valid_actions = self.env.get_valid_actions(self.human_player)
            
            if not valid_actions:
                messagebox.showinfo("Hint", "No valid moves available!")
                return
            
            # Use agent to suggest best move
            suggested_action = self.agent.select_action(state, valid_actions, training=False)
            from_r, from_c, to_r, to_c = self.env._decode_action(suggested_action)
            
            hint_text = f"Suggested move: ({from_r},{from_c}) → ({to_r},{to_c})"
            messagebox.showinfo("Hint", hint_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate hint: {str(e)}")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

class StrategoConsole:
    """Console interface for playing Stratego"""
    
    def __init__(self):
        self.env = StrategoEnv()
        self.agent = None
        
    def load_agent(self, model_path):
        """Load trained agent"""
        try:
            state_size = self.env.get_state_space_size()
            action_size = self.env.get_action_space_size()
            
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                player_id=1,
                epsilon=0.0
            )
            
            if self.agent.load_model(model_path):
                self.agent.set_training_mode(False)
                print("Agent loaded successfully!")
                return True
            else:
                print("Failed to load agent")
                return False
                
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
    
    def play_game(self):
        """Play a game in console"""
        if self.agent is None:
            print("Please load an agent first")
            return
        
        print("Starting new game! You are Red (bottom), AI is Blue (top)")
        print("Enter moves as: from_row,from_col to_row,to_col (e.g., '6,0 5,0')")
        print("Type 'quit' to exit, 'help' for commands\n")
        
        state = self.env.reset()
        human_player = 0  # Human is red
        
        while not self.env.game_over:
            self.env.render()
            
            if self.env.current_player == human_player:
                # Human turn
                valid_actions = self.env.get_valid_actions(human_player)
                if not valid_actions:
                    print("No valid moves available!")
                    break
                
                while True:
                    try:
                        user_input = input("Your move: ").strip().lower()
                        
                        if user_input == 'quit':
                            return
                        elif user_input == 'help':
                            self.print_help()
                            continue
                        elif user_input == 'hint' and self.agent:
                            self.give_hint(human_player)
                            continue
                        
                        # Parse move
                        parts = user_input.split()
                        if len(parts) != 2:
                            print("Invalid format. Use: from_row,from_col to_row,to_col")
                            continue
                        
                        from_pos = parts[0].split(',')
                        to_pos = parts[1].split(',')
                        
                        if len(from_pos) != 2 or len(to_pos) != 2:
                            print("Invalid format. Use: from_row,from_col to_row,to_col")
                            continue
                        
                        from_r, from_c = int(from_pos[0]), int(from_pos[1])
                        to_r, to_c = int(to_pos[0]), int(to_pos[1])
                        
                        action = self.env._encode_action(from_r, from_c, to_r, to_c)
                        
                        if action in valid_actions:
                            state, reward, done, info = self.env.step(action)
                            print(f"Move executed. Reward: {reward}")
                            break
                        else:
                            print("Invalid move. Try again.")
                    
                    except (ValueError, IndexError):
                        print("Invalid input. Use format: from_row,from_col to_row,to_col")
                    except Exception as e:
                        print(f"Error: {e}")
            
            else:
                # AI turn
                print("AI is thinking...")
                valid_actions = self.env.get_valid_actions(1)
                
                if not valid_actions:
                    print("AI has no valid moves!")
                    break
                
                action = self.agent.select_action(state, valid_actions, training=False)
                state, reward, done, info = self.env.step(action)
                
                from_r, from_c, to_r, to_c = self.env._decode_action(action)
                print(f"AI moved: ({from_r},{from_c}) → ({to_r},{to_c})")
        
        # Game over
        self.env.render()
        if self.env.winner is not None:
            winner = "You" if self.env.winner == human_player else "AI"
            print(f"\nGame Over! {winner} won!")
        else:
            print("\nGame Over! It's a draw!")
    
    def print_help(self):
        """Print help information"""
        print("\nCommands:")
        print("  move: from_row,from_col to_row,to_col (e.g., '6,0 5,0')")
        print("  hint: Get AI suggestion for your move")
        print("  help: Show this help")
        print("  quit: Exit game")
        print("\nPiece abbreviations:")
        print("  F=Flag, S=Spy, 2-10=Ranks, B=Bomb")
        print("  Red pieces are yours, Blue are AI's")
        print()
    
    def give_hint(self, player):
        """Give hint to human player"""
        try:
            state = self.env.get_state()
            valid_actions = self.env.get_valid_actions(player)
            
            if not valid_actions:
                print("No valid moves available!")
                return
            
            suggested_action = self.agent.select_action(state, valid_actions, training=False)
            from_r, from_c, to_r, to_c = self.env._decode_action(suggested_action)
            
            print(f"Hint: Try moving from ({from_r},{from_c}) to ({to_r},{to_c})")
            
        except Exception as e:
            print(f"Could not generate hint: {e}")

def main():
    """Main function to choose interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Stratego against trained DQN agent')
    parser.add_argument('--interface', choices=['gui', 'console'], default='gui',
                       help='Choose interface type')
    parser.add_argument('--model', type=str, help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.interface == 'gui':
        app = StrategoGUI()
        if args.model:
            app.model_path_var.set(args.model)
            app.load_model()
        app.run()
    else:
        app = StrategoConsole()
        if args.model:
            app.load_agent(args.model)
        app.play_game()

if __name__ == "__main__":
    main()