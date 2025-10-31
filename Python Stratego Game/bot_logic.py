import torch
import torch.nn as nn
import numpy as np

class StrategoNet(nn.Module):
    def __init__(self):
        super(StrategoNet, self).__init__()
        self.fc1 = nn.Linear(200, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1000)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

class BotLogic:
    def __init__(self, model_path):
        self.model = StrategoNet()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['q_network_state_dict'])
        self.model.eval()

    def choose_move(self, board, owner):
        board_state = self.get_board_state(board, owner)
        with torch.no_grad():
            output = self.model(board_state)
            legal_moves = self.get_legal_moves_mask(board, owner, output.shape[-1])
            masked_output = output * legal_moves
            move_index = torch.argmax(masked_output).item()
        
        return self.index_to_move(move_index, board, owner)

    def get_board_state(self, board, owner):
        # The model expects a 200-element vector.
        # We'll create two 10x10 boards (one for each player) and flatten them.
        player_state = np.zeros((10, 10), dtype=np.float32)
        opponent_state = np.zeros((10, 10), dtype=np.float32)
        
        for r in range(10):
            for c in range(10):
                piece = board.get((r, c))
                if piece:
                    if piece.owner == owner:
                        player_state[r, c] = piece.rank
                    else:
                        # Hide opponent piece ranks unless they are revealed
                        opponent_state[r, c] = piece.rank if piece.revealed else 1

        # Flatten and concatenate the two boards to create a 200-element vector
        state_vector = np.concatenate(
            (player_state.flatten(), opponent_state.flatten()), axis=0
        )
        return torch.from_numpy(state_vector).unsqueeze(0)

    def get_legal_moves_mask(self, board, owner, output_size):
        mask = torch.zeros(output_size)
        moves = []
        for src in board.owner_positions(owner):
            piece = board.get(src)
            if not piece or not piece.is_movable():
                continue
            legal_dsts = board.legal_moves_from(src)
            for dst in legal_dsts:
                moves.append((src, dst))
                move_index = self.move_to_index(src, dst)
                if move_index < output_size:
                    mask[move_index] = 1.0
        return mask

    def move_to_index(self, src, dst):
        # Action space: 100 source squares * 10 destination rows = 1000
        src_index = src[0] * 10 + src[1]
        dst_row = dst[0]
        return src_index * 10 + dst_row

    def get_direction_index(self, src, dst):
        dr, dc = dst[0] - src[0], dst[1] - src[1]
        if dr == -1 and dc == 0: return 0  # Up
        if dr == 1 and dc == 0: return 1   # Down
        if dr == 0 and dc == -1: return 2  # Left
        if dr == 0 and dc == 1: return 3   # Right
        return 0

    def index_to_move(self, index, board, owner):
        src_index = index // 10
        dst_row = index % 10
        src_r, src_c = src_index // 10, src_index % 10
        src = (src_r, src_c)

        # The model only predicts the destination row. We must find the destination column.
        # This assumes a piece can only move to one valid square in a given row.
        legal_moves = board.legal_moves_from(src)
        for move in legal_moves:
            if move[0] == dst_row:
                return (src, move)

        # Fallback if the predicted move is illegal or ambiguous
        dst = (dst_row, src_c) # A guess, but might be illegal

        legal_moves = board.legal_moves_from(src)
        if dst in legal_moves:
            return (src, dst)
        
        # Fallback to a random legal move if the chosen move is invalid
        all_moves = []
        for s in board.owner_positions(owner):
            if board.get(s) and board.get(s).is_movable():
                all_moves.extend([(s, d) for d in board.legal_moves_from(s)])
        return random.choice(all_moves) if all_moves else None
