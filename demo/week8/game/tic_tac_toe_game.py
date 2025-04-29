# tic_tac_toe_game.py

from typing import List, Tuple, Optional, Dict, Any
import random
import matplotlib.pyplot as plt
import time

class TicTacToeGame:
    def __init__(self, player_x, player_o, delay_seconds: float = 1.0):
        self.board = [['_' for _ in range(3)] for _ in range(3)]
        self.player_x = player_x
        self.player_o = player_o
        self.current_player = player_x
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.delay_seconds = delay_seconds

    def get_board_state(self):
        return [row.copy() for row in self.board]

    def display_board(self, with_coords: bool = True):
        """Display the current board state with matplotlib for better visualization."""
        fig, ax = plt.subplots(figsize=(3, 3))
        
        # Fill background color (optional, looks cleaner)
        ax.set_facecolor("#f9f9f9")
        
        # Draw grid lines
        for i in range(1, 3):
            ax.axhline(i, color='black', linewidth=1.5)
            ax.axvline(i, color='black', linewidth=1.5)
        
        # Draw markers
        for row in range(3):
            for col in range(3):
                x = col + 0.5
                y = row + 0.5
                if self.board[row][col] == 'X':
                    ax.text(x, y, 'X', fontsize=30, ha='center', va='center', fontweight='bold', color='#0074D9')
                elif self.board[row][col] == 'O':
                    ax.text(x, y, 'O', fontsize=30, ha='center', va='center', fontweight='bold', color='#FF4136')
                elif with_coords:
                    ax.text(x, y, f"{row},{col}", fontsize=12, ha='center', va='center', color='black')

        # Set board limits and remove ticks
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')  # Make sure the board is a square
        ax.invert_yaxis()  # (0,0) on top-left like traditional

        # Add a subtle border
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # Title
        if self.game_over:
            if self.winner:
                title = f"{self.winner.name} ({self.winner.marker}) wins!"
            else:
                title = "Draw!"
        else:
            title = f"{self.current_player.name}'s turn ({self.current_player.marker})"
        plt.title(title, fontsize=14, pad=10)

        # Adjust layout
        plt.tight_layout(pad=1.5)
        plt.show()

    def print_board(self, with_coords: bool = True):
        print("\nCurrent board:")
        if with_coords:
            print("  0 1 2")
            for i, row in enumerate(self.board):
                print(f"{i} {' '.join(row)}")
        else:
            for row in self.board:
                print(" ".join(row))
        print()

    def make_move(self, row: int, col: int) -> bool:
        if self.game_over or not (0 <= row <= 2 and 0 <= col <= 2) or self.board[row][col] != '_':
            return False
        self.board[row][col] = self.current_player.marker
        self.move_history.append((self.current_player.name, row, col))
        return True

    def check_winner(self) -> Optional[str]:
        lines = (
            self.board,
            zip(*self.board),
            [[self.board[i][i] for i in range(3)]],
            [[self.board[i][2-i] for i in range(3)]],
        )
        for group in lines:
            for line in group:
                if line[0] != '_' and all(cell == line[0] for cell in line):
                    return line[0]
        if all(cell != '_' for row in self.board for cell in row):
            return "Draw"
        return None

    def switch_player(self):
        self.current_player = self.player_o if self.current_player == self.player_x else self.player_x

    def next_move(self) -> Dict[str, Any]:
        """Play one move (step-by-step interactive mode)."""
        if self.game_over:
            return {
                "status": "game_over",
                "winner": self.winner.name if self.winner else "Draw",
                "board": self.get_board_state(),
                "move_history": self.move_history
            }

        valid_moves = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == '_']
        print(f"Valid moves: {valid_moves}")

        row, col = self.current_player.get_move(self.board)
        player_name = self.current_player.name
        player_marker = self.current_player.marker

        if self.make_move(row, col):
            print(f"{player_name} ({player_marker}) placed at [{row}, {col}]")
        else:
            print(f"{player_name} ({player_marker}) tried invalid move [{row}, {col}]. Choosing random fallback...")
            if valid_moves:
                row, col = random.choice(valid_moves)
                self.make_move(row, col)
                print(f"{player_name} ({player_marker}) fallback random move to [{row}, {col}]")

        result = self.check_winner()
        if result:
            self.game_over = True
            self.winner = None if result == "Draw" else self.current_player

        if not self.game_over:
            self.switch_player()

        return {
            "status": "game_over" if self.game_over else "in_progress",
            "current_player": None if self.game_over else self.current_player.name,
            "winner": self.winner.name if self.game_over and self.winner else "Draw" if self.game_over else None,
            "board": self.get_board_state(),
            "last_move": self.move_history[-1],
            "move_history": self.move_history
        }

    def play_game(self, verbose: bool = True, use_display: bool = False) -> Tuple[Optional[Any], List[Tuple[str, int, int]]]:
        """Play the full game automatically."""
        if verbose:
            print(f"Starting a new game: {self.player_x.name} (X) vs {self.player_o.name} (O)")
            if use_display:
                self.display_board()
            else:
                self.print_board()

        while not self.game_over:
            valid_moves = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == '_']
            if verbose:
                print(f"Valid moves: {valid_moves}")

            row, col = self.current_player.get_move(self.board)
            player_name = self.current_player.name
            player_marker = self.current_player.marker

            if self.make_move(row, col):
                if verbose:
                    print(f"{player_name} ({player_marker}) placed at [{row}, {col}]")
            else:
                if verbose:
                    print(f"{player_name} ({player_marker}) tried invalid move [{row}, {col}]. Choosing random fallback...")
                if valid_moves:
                    row, col = random.choice(valid_moves)
                    self.make_move(row, col)
                    if verbose:
                        print(f"{player_name} ({player_marker}) fallback random move to [{row}, {col}]")

            if verbose:
                if use_display:
                    self.display_board()
                else:
                    self.print_board()

            result = self.check_winner()
            if result:
                self.game_over = True
                self.winner = None if result == "Draw" else self.current_player
                break

            self.switch_player()

            if self.delay_seconds > 0 and self.current_player.__class__.__name__ != "HumanPlayer":
                time.sleep(self.delay_seconds)

        return self.winner, self.move_history
    
    def print_game_summary(self):
        """Print a clean move history and winner after the game."""
        print("\nGame Summary:")
        for idx, (player_name, row, col) in enumerate(self.move_history):
            print(f"{idx+1}. {player_name} -> [{row},{col}]")
        
        if self.game_over:
            if self.winner:
                print(f"\n🏆 Winner: {self.winner.name} ({self.winner.marker})")
            else:
                print("\n🤝 It's a draw!")
        else:
            print("\nGame not finished yet.")

def create_game(player_x, player_o, delay_seconds=1.0) -> TicTacToeGame:
    return TicTacToeGame(player_x, player_o, delay_seconds)