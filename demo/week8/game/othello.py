import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import sys

# API keys configuration - set your keys here directly
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"

# Abstract base class for all players (LLMs and humans)
class Player:
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color  # 'B' for black or 'W' for white
    
    def get_move(self, board: List[List[str]], valid_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Get the next move from the player."""
        raise NotImplementedError("Subclasses must implement this method")


# Human player that takes input from the terminal
class HumanPlayer(Player):
    def __init__(self, name: str, color: str):
        super().__init__(name, color)
    
    def get_move(self, board: List[List[str]], valid_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Get the next move from the human player via terminal input."""
        print(f"\n{self.name}'s turn ({self.color})")
        self._print_board(board)
        
        if not valid_moves:
            print("No valid moves available. Your turn will be skipped.")
            return None
        
        print("Valid moves:", valid_moves)
        
        while True:
            try:
                move_input = input(f"Enter your move as 'row,col' (0-3) or 'skip' to pass: ")
                
                if move_input.lower() == 'skip' or move_input.lower() == 'pass':
                    return None
                
                # Parse the input
                parts = move_input.strip().split(',')
                if len(parts) != 2:
                    raise ValueError("Input must be in the format 'row,col'")
                
                row, col = int(parts[0]), int(parts[1])
                
                # Validate the move
                if not (0 <= row <= 3 and 0 <= col <= 3):
                    print("Invalid move: Position must be between 0 and 3")
                    continue
                
                if (row, col) not in valid_moves:
                    print("Invalid move: This is not a valid position to place your piece")
                    continue
                
                return row, col
                
            except ValueError as e:
                print(f"Invalid input: {str(e)}. Try again.")
    
    def _print_board(self, board: List[List[str]]):
        """Print the current state of the board."""
        print("\nCurrent board:")
        print("  0 1 2 3")
        for i, row in enumerate(board):
            print(f"{i} {' '.join(row)}")
        print()


# LLM player class that handles multiple LLM providers
class LLMPlayer(Player):
    def __init__(self, 
                 name: str, 
                 color: str, 
                 model: str, 
                 api_key: str, 
                 provider: Literal["gemini", "openai", "anthropic"] = "gemini"):
        super().__init__(name, color)
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.chat_history = []
        self.system_prompt = ""
        
        # Configure API endpoints and parameters based on provider
        if provider == "gemini":
            self.endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            self.params = {"key": api_key}
            self.headers = {"Content-Type": "application/json"}
        elif provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.params = {}
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        elif provider == "anthropic":
            self.endpoint = "https://api.anthropic.com/v1/messages"
            self.params = {}
            self.headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self._initialize_chat()
    
    def _initialize_chat(self):
        """Initialize the chat with instructions."""
        self.system_prompt = """
        You are playing a game of 4x4 Reversi (also known as Othello). The board is represented as a 4x4 grid. 
        Empty spaces are represented by '.', your pieces are represented by '{color}', 
        and your opponent's pieces are represented by '{opponent_color}'.
        
        The rules of Reversi:
        1. You place your piece on the board, but only in a position that 'outflanks' at least one of your opponent's pieces.
        2. Outflanking means placing your piece such that one or more of your opponent's pieces are in a straight line 
           (horizontal, vertical, or diagonal) between your newly placed piece and another of your pieces already on the board.
        3. When you outflank one or more opponent pieces, those pieces are flipped to your color.
        4. If you cannot make a valid move, your turn is skipped.
        5. The game ends when the board is full or neither player can make a move.
        6. The player with the most pieces of their color on the board wins.
        
        The board positions are as follows:
        [0,0] [0,1] [0,2] [0,3]
        [1,0] [1,1] [1,2] [1,3]
        [2,0] [2,1] [2,2] [2,3]
        [3,0] [3,1] [3,2] [3,3]
        
        When it's your turn, analyze the board and provide your next move in the format: [row, column]
        For example, if you want to place your piece in the top-right corner, respond with: [0, 3]
        
        Choose your moves strategically to control the board and flip as many of your opponent's pieces as possible.
        If there are no valid moves, respond with: "skip"
        
        Respond ONLY with your move coordinates in the format [row, column] or "skip" - no explanation or other text.
        """.format(color=self.color, opponent_color='W' if self.color == 'B' else 'B')
        
        # For OpenAI and Anthropic, we can use the system role
        if self.provider in ["openai", "anthropic"]:
            self.update_history("system", self.system_prompt)
        else:
            # For Gemini, we'll add the system prompt as a user message since it doesn't support system role
            first_prompt = f"Instructions for playing Reversi:\n{self.system_prompt}\nDo you understand these instructions? Remember to only respond with move coordinates in the format [row, column] or 'skip'."
            self.update_history("user", first_prompt)
            self.update_history("assistant", "I understand. I'll respond with only the move coordinates in the format [row, column] or 'skip' if there are no valid moves.")
    
    def update_history(self, role: str, content: str):
        """Add a message to the chat history."""
        # For Gemini, we skip system messages since they're not supported
        if self.provider == "gemini" and role == "system":
            return
            
        self.chat_history.append({"role": role, "content": content})
    
    def _format_board_for_prompt(self, board: List[List[str]], valid_moves: List[Tuple[int, int]]) -> str:
        """Format the board as a string for the prompt."""
        # Convert 'B' and 'W' to 'B' and 'W' for better readability
        formatted_board = "\nCurrent board:\n"
        formatted_board += "  0 1 2 3\n"
        for i, row in enumerate(board):
            formatted_board += f"{i} "
            for cell in row:
                formatted_board += cell + " "
            formatted_board += "\n"
        
        # Add valid moves information
        formatted_board += "\nYour color: " + self.color + "\n"
        formatted_board += "Valid moves: " + str(valid_moves) + "\n"
        
        return formatted_board
    
    def _format_request_data(self, prompt: str):
        """Format the request data according to the provider's API."""
        if self.provider == "gemini":
            # Gemini doesn't support system messages, so we only include user/model messages
            filtered_history = [msg for msg in self.chat_history if msg["role"] != "system"]
            return {
                "contents": [
                    {"role": "user" if msg["role"] == "user" else "model", 
                     "parts": [{"text": msg["content"]}]} 
                    for msg in filtered_history
                ]
            }
        elif self.provider == "openai":
            return {
                "model": self.model,
                "messages": self.chat_history,
                "temperature": 0.5,
                "max_tokens": 20
            }
        elif self.provider == "anthropic":
            # For Anthropic (Claude), we need to handle 'system' message differently
            system_message = next((msg["content"] for msg in self.chat_history if msg["role"] == "system"), None)
            
            messages = []
            for msg in self.chat_history:
                if msg["role"] != "system":  # Skip system message as it's handled separately
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            return {
                "model": self.model,
                "messages": messages,
                "system": system_message,
                "max_tokens": 20
            }
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Extract the response text from the API response data."""
        if self.provider == "gemini":
            return response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        elif self.provider == "openai":
            return response_data["choices"][0]["message"]["content"].strip()
        elif self.provider == "anthropic":
            return response_data["content"][0]["text"].strip()
    
    def get_move(self, board: List[List[str]], valid_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Get the next move from the LLM model."""
        if not valid_moves:
            return None  # No valid moves available, skip turn
            
        board_str = self._format_board_for_prompt(board, valid_moves)
        prompt = f"{board_str}\nIt's your turn. Where would you like to place your {self.color} piece? Respond only with [row, column] or 'skip'."
        
        self.update_history("user", prompt)
        
        # Prepare the request data based on the provider
        data = self._format_request_data(prompt)
        
        # Make the API call
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                params=self.params,
                json=data,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Error with {self.provider} API ({response.status_code}): {response.text}")
                return self._get_random_valid_move(valid_moves)
            
            response_data = response.json()
            response_text = self._parse_response(response_data)
            
            # Add the response to chat history
            self.update_history("assistant", response_text)
            
            # Check for "skip" response
            if response_text.lower().strip() in ["skip", "pass", "no valid moves"]:
                return None
            
            # Parse the move from the response
            import re
            match = re.search(r'\[(\d+)\s*,\s*(\d+)\]', response_text)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                if (row, col) in valid_moves:
                    return row, col
            
            # If we couldn't parse a valid move, fall back to random
            print(f"Could not parse valid move from: {response_text}")
            return self._get_random_valid_move(valid_moves)
            
        except Exception as e:
            print(f"Error with {self.provider} API request: {str(e)}")
            return self._get_random_valid_move(valid_moves)
    
    def _get_random_valid_move(self, valid_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Get a random valid move as a fallback."""
        if not valid_moves:
            return None
        
        return random.choice(valid_moves)


class ReversiGame:
    def __init__(self, player_black: Player, player_white: Player, delay_seconds: float = 1.0):
        # Initialize empty 4x4 board with '.' for empty cells
        self.board = [['.' for _ in range(4)] for _ in range(4)]
        
        # In 4x4 Reversi, starting position has 4 pieces in the center
        self.board[1][1] = 'W'
        self.board[1][2] = 'B'
        self.board[2][1] = 'B'
        self.board[2][2] = 'W'
        
        self.player_black = player_black  # Black ('B') goes first
        self.player_white = player_white
        self.current_player = player_black
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.delay_seconds = delay_seconds
        self.consecutive_skips = 0
    
    def get_board_state(self):
        """Return a copy of the current board state."""
        return [row.copy() for row in self.board]
    
    def display_board(self, with_coords: bool = True, with_valid_moves: bool = True):
        """Display the current board state with matplotlib for better visualization."""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Fill background color
        ax.set_facecolor("#008000")  # Green background like a game board
        
        # Draw grid lines
        for i in range(1, 4):
            ax.axhline(i, color='black', linewidth=1.5)
            ax.axvline(i, color='black', linewidth=1.5)
        
        # Calculate valid moves for the current player
        valid_moves = self.get_valid_moves(self.current_player.color) if with_valid_moves and not self.game_over else []
        
        # Draw markers
        for row in range(4):
            for col in range(4):
                x = col + 0.5
                y = row + 0.5
                if self.board[row][col] == 'B':
                    # Black piece
                    circle = plt.Circle((x, y), 0.4, color='black', fill=True)
                    ax.add_patch(circle)
                elif self.board[row][col] == 'W':
                    # White piece
                    circle = plt.Circle((x, y), 0.4, color='white', fill=True, edgecolor='black')
                    ax.add_patch(circle)
                elif with_coords and (row, col) not in valid_moves:
                    ax.text(x, y, f"{row},{col}", fontsize=10, ha='center', va='center', color='black')
                
                # Highlight valid moves
                if with_valid_moves and (row, col) in valid_moves:
                    circle = plt.Circle((x, y), 0.4, color='lightgray', fill=True, alpha=0.5)
                    ax.add_patch(circle)
                    ax.text(x, y, f"{row},{col}", fontsize=10, ha='center', va='center', color='black')

        # Set board limits and remove ticks
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')  # Make sure the board is a square
        ax.invert_yaxis()  # (0,0) on top-left
        
        # Add a subtle border
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # Count pieces
        black_count = sum(row.count('B') for row in self.board)
        white_count = sum(row.count('W') for row in self.board)
        
        # Title
        if self.game_over:
            if self.winner:
                title = f"{self.winner.name} ({self.winner.color}) wins: B {black_count}-{white_count} W"
            else:
                title = f"Draw: B {black_count}-{white_count} W"
        else:
            title = f"{self.current_player.name}'s turn ({self.current_player.color}): B {black_count}-{white_count} W"
            if not valid_moves:
                title += " (No valid moves)"
        
        plt.title(title, fontsize=14, pad=10)

        # Adjust layout
        plt.tight_layout(pad=1.5)
        plt.show()
    
    def print_board(self, with_coords: bool = True):
        """Print the current state of the board as text."""
        # Count pieces
        black_count = sum(row.count('B') for row in self.board)
        white_count = sum(row.count('W') for row in self.board)
        
        print("\nCurrent board:")
        if with_coords:
            print("  0 1 2 3")
            for i, row in enumerate(self.board):
                print(f"{i} {' '.join(row)}")
        else:
            for row in self.board:
                print(" ".join(row))
        
        print(f"\nScore: Black: {black_count}, White: {white_count}")
        
        # Show valid moves for current player if game is not over
        if not self.game_over:
            valid_moves = self.get_valid_moves(self.current_player.color)
            if valid_moves:
                print(f"Valid moves for {self.current_player.color}: {valid_moves}")
            else:
                print(f"No valid moves for {self.current_player.color}")
        
        print()
    
    def get_valid_moves(self, color: str) -> List[Tuple[int, int]]:
        """Return a list of valid moves for the given color."""
        valid_moves = []
        
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == '.':  # Empty cell
                    # Check if this is a valid move
                    if self._would_flip_any(row, col, color):
                        valid_moves.append((row, col))
        
        return valid_moves
    
    def _would_flip_any(self, row: int, col: int, color: str) -> bool:
        """Check if placing a piece at (row, col) would flip any opponent pieces."""
        opponent_color = 'W' if color == 'B' else 'B'
        
        # Check all 8 directions (clockwise from north)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                     (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        for dr, dc in directions:
            # Check this direction
            r, c = row + dr, col + dc
            
            # Skip if the adjacent cell is not opponent's color
            if not (0 <= r < 4 and 0 <= c < 4 and self.board[r][c] == opponent_color):
                continue
            
            # Continue in this direction
            r += dr
            c += dc
            
            # Look for player's color after opponent's color(s)
            while 0 <= r < 4 and 0 <= c < 4:
                if self.board[r][c] == '.':  # Empty cell
                    break
                if self.board[r][c] == color:  # Found player's color
                    return True
                # Continue in the same direction
                r += dr
                c += dc
        
        return False
    
    def _get_flipped_pieces(self, row: int, col: int, color: str) -> List[Tuple[int, int]]:
        """Get a list of pieces that would be flipped if a piece is placed at (row, col)."""
        opponent_color = 'W' if color == 'B' else 'B'
        flipped = []
        
        # Check all 8 directions
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                     (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        for dr, dc in directions:
            # Pieces to flip in this direction
            direction_flipped = []
            
            # Check this direction
            r, c = row + dr, col + dc
            
            # Skip if the adjacent cell is not opponent's color
            if not (0 <= r < 4 and 0 <= c < 4 and self.board[r][c] == opponent_color):
                continue
            
            # Remember this opponent piece
            direction_flipped.append((r, c))
            
            # Continue in this direction
            r += dr
            c += dc
            
            # Look for player's color after opponent's color(s)
            while 0 <= r < 4 and 0 <= c < 4:
                if self.board[r][c] == '.':  # Empty cell
                    direction_flipped = []  # Invalid, clear the list
                    break
                if self.board[r][c] == color:  # Found player's color
                    flipped.extend(direction_flipped)  # Add all pieces in this direction
                    break
                # Add this opponent piece and continue
                direction_flipped.append((r, c))
                r += dr
                c += dc
        
        return flipped
    
    def make_move(self, row: int, col: int) -> bool:
        """Attempt to make a move at the specified position."""
        if self.game_over:
            print("Game is already over!")
            return False
        
        color = self.current_player.color
        
        if not (0 <= row < 4 and 0 <= col < 4):
            print(f"Invalid move: [{row}, {col}] is out of bounds")
            return False
        
        if self.board[row][col] != '.':
            print(f"Invalid move: [{row}, {col}] is already occupied")
            return False
        
        # Get pieces that would be flipped by this move
        flipped = self._get_flipped_pieces(row, col, color)
        
        if not flipped:
            print(f"Invalid move: [{row}, {col}] doesn't flip any pieces")
            return False
        
        # Valid move - place the piece and flip opponent's pieces
        self.board[row][col] = color
        for r, c in flipped:
            self.board[r][c] = color
        
        self.move_history.append((self.current_player.name, row, col, len(flipped)))
        self.consecutive_skips = 0  # Reset skip counter
        
        return True
    
    def skip_turn(self):
        """Skip the current player's turn."""
        self.move_history.append((self.current_player.name, None, None, 0))  # Record the skip
        self.consecutive_skips += 1
    
    def check_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over if board is full
        if all(cell != '.' for row in self.board for cell in row):
            return True
        
        # Game is over if neither player can make a move (two consecutive skips)
        if self.consecutive_skips >= 2:
            return True
        
        return False
    
    def determine_winner(self) -> Optional[Player]:
        """Determine the winner based on piece count."""
        black_count = sum(row.count('B') for row in self.board)
        white_count = sum(row.count('W') for row in self.board)
        
        if black_count > white_count:
            return self.player_black
        elif white_count > black_count:
            return self.player_white
        else:
            return None  # Draw
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player = self.player_white if self.current_player == self.player_black else self.player_black
    
    def next_move(self) -> Dict[str, Any]:
        """Execute the next move and return the result (for interactive mode)."""
        if self.game_over:
            return {
                "status": "game_over",
                "winner": self.winner.name if self.winner else "Draw",
                "board": self.get_board_state(),
                "move_history": self.move_history
            }
        
        # Get valid moves for current player
        valid_moves = self.get_valid_moves(self.current_player.color)
        
        # Get move from current player
        player_name = self.current_player.name
        player_color = self.current_player.color
        
        print(f"{player_name}'s turn ({player_color})...")
        
        # Check if player can make a move
        if not valid_moves:
            print(f"No valid moves for {player_name}. Turn skipped.")
            self.skip_turn()
            
            # Check if game is over
            if self.check_game_over():
                self.game_over = True
                self.winner = self.determine_winner()
                
                if self.winner:
                    print(f"Game over! {self.winner.name} ({self.winner.color}) wins!")
                else:
                    print("Game over! It's a draw!")
            else:
                # Switch to the other player
                self.switch_player()
            
            # Always display the board after a skip
            self.print_board()
            
            # Return current game state
            return {
                "status": "game_over" if self.game_over else "in_progress",
                "current_player": None if self.game_over else self.current_player.name,
                "winner": self.winner.name if self.game_over and self.winner else "Draw" if self.game_over else None,
                "board": self.get_board_state(),
                "last_move": self.move_history[-1] if self.move_history else None,
                "move_history": self.move_history
            }
        
        # Get and execute the move
        move = self.current_player.get_move(self.board, valid_moves)
        
        if move is None:
            # Player chose to skip even though valid moves exist
            print(f"{player_name} skips their turn.")
            self.skip_turn()
        else:
            row, col = move
            print(f"{player_name} chooses: [{row}, {col}]")
            
            # Make the move
            if self.make_move(row, col):
                print(f"Move successful. Flipped {self.move_history[-1][3]} pieces.")
            else:
                # Invalid move, try random valid move as fallback
                print(f"Invalid move from {player_name}. Using random valid move instead...")
                if valid_moves:
                    row, col = random.choice(valid_moves)
                    self.make_move(row, col)
                    print(f"Used random move: [{row}, {col}], flipped {self.move_history[-1][3]} pieces.")
                else:
                    self.skip_turn()
                    print(f"No valid moves available. {player_name}'s turn skipped.")
        
        # Always display the board after a move or skip in interactive mode
        self.print_board()
        
        # Check if game is over
        if self.check_game_over():
            self.game_over = True
            self.winner = self.determine_winner()
            
            if self.winner:
                print(f"Game over! {self.winner.name} ({self.winner.color}) wins!")
            else:
                print("Game over! It's a draw!")
        else:
            # Switch to the other player
            self.switch_player()
        
        # Return current game state
        return {
            "status": "game_over" if self.game_over else "in_progress",
            "current_player": None if self.game_over else self.current_player.name,
            "winner": self.winner.name if self.game_over and self.winner else "Draw" if self.game_over else None,
            "board": self.get_board_state(),
            "last_move": self.move_history[-1] if self.move_history else None,
            "move_history": self.move_history
        }
    
    def play_game(self, verbose: bool = True, use_display: bool = False) -> Tuple[Optional[Player], List[Tuple]]:
        """Play a full game of Reversi (for complete mode)."""
        if verbose:
            print(f"Starting a new game: {self.player_black.name} (B) vs {self.player_white.name} (W)")
            if use_display:
                self.display_board()
            else:
                self.print_board()
        
        while not self.game_over:
            # Get valid moves for current player
            valid_moves = self.get_valid_moves(self.current_player.color)
            
            # Get move from current player
            player_name = self.current_player.name
            player_color = self.current_player.color
            
            if verbose:
                print(f"\n{player_name}'s turn ({player_color})...")
            
            # Check if player can make a move
            if not valid_moves:
                if verbose:
                    print(f"No valid moves for {player_name}. Turn skipped.")
                self.skip_turn()
                
                # Check if game is over
                if self.check_game_over():
                    self.game_over = True
                    self.winner = self.determine_winner()
                    
                    if verbose:
                        if self.winner:
                            print(f"Game over! {self.winner.name} ({self.winner.color}) wins!")
                        else:
                            print("Game over! It's a draw!")
                else:
                    # Switch to the other player
                    self.switch_player()
                
                # Add a small delay between moves for better visualization
                if self.delay_seconds > 0 and verbose and not isinstance(self.current_player, HumanPlayer):
                    time.sleep(self.delay_seconds)
                
                # Show board after skipping
                if verbose and use_display:
                    if 'ipykernel' in sys.modules:  # Check if running in notebook
                        clear_output(wait=True)  # Clear previous board
                    self.display_board()
                elif verbose:
                    self.print_board()
                    
                continue
            
            # Get and execute the move
            move = self.current_player.get_move(self.board, valid_moves)
            
            if move is None:
                # Player chose to skip even though valid moves exist
                if verbose:
                    print(f"{player_name} skips their turn.")
                self.skip_turn()
                
                # Check if game is over after skipping
                if self.check_game_over():
                    self.game_over = True
                    self.winner = self.determine_winner()
                    
                    if verbose:
                        if self.winner:
                            print(f"Game over! {self.winner.name} ({self.winner.color}) wins!")
                        else:
                            print("Game over! It's a draw!")
                else:
                    # Switch to the other player
                    self.switch_player()
                
                # Add a small delay between moves for better visualization
                if self.delay_seconds > 0 and verbose and not isinstance(self.current_player, HumanPlayer):
                    time.sleep(self.delay_seconds)
                
                # Show board after skipping
                if verbose and use_display:
                    if 'ipykernel' in sys.modules:  # Check if running in notebook
                        clear_output(wait=True)  # Clear previous board
                    self.display_board()
                elif verbose:
                    self.print_board()
                    
                continue
            
            # Try to make the move
            row, col = move
            
            if verbose:
                print(f"{player_name} chooses: [{row}, {col}]")
            
            # Make the move
            if self.make_move(row, col):
                if verbose:
                    print(f"Move successful. Flipped {self.move_history[-1][3]} pieces.")
                    
                    if use_display:
                        if 'ipykernel' in sys.modules:  # Check if running in notebook
                            clear_output(wait=True)  # Clear previous board
                        self.display_board()
                    else:
                        self.print_board()
                
                # Check if game is over
                if self.check_game_over():
                    self.game_over = True
                    self.winner = self.determine_winner()
                    
                    if verbose:
                        if self.winner:
                            print(f"Game over! {self.winner.name} ({self.winner.color}) wins!")
                        else:
                            print("Game over! It's a draw!")
                else:
                    # Switch to the other player
                    self.switch_player()
                    
                    # Add a small delay between moves for better visualization
                    if self.delay_seconds > 0 and verbose and not isinstance(self.current_player, HumanPlayer):
                        time.sleep(self.delay_seconds)
            else:
                # Invalid move, try random valid move as fallback (if not human)
                if verbose:
                    print(f"Invalid move from {player_name}. Trying again...")
                
                # If it's an LLM player, use a random valid move
                if not isinstance(self.current_player, HumanPlayer):
                    if valid_moves:
                        row, col = random.choice(valid_moves)
                        if verbose:
                            print(f"Using random move: [{row}, {col}]")
                        self.make_move(row, col)
                        
                        if verbose:
                            print(f"Flipped {self.move_history[-1][3]} pieces.")
                            if use_display:
                                if 'ipykernel' in sys.modules:
                                    clear_output(wait=True)
                                self.display_board()
                            else:
                                self.print_board()
                        
                        # Check if game is over
                        if self.check_game_over():
                            self.game_over = True
                            self.winner = self.determine_winner()
                            
                            if verbose:
                                if self.winner:
                                    print(f"Game over! {self.winner.name} ({self.winner.color}) wins!")
                                else:
                                    print("Game over! It's a draw!")
                        else:
                            # Switch to the other player
                            self.switch_player()
                            
                            # Add a small delay between moves for better visualization
                            if self.delay_seconds > 0 and verbose and not isinstance(self.current_player, HumanPlayer):
                                time.sleep(self.delay_seconds)
        
        # Return the winner and move history at the end of the game
        return self.winner, self.move_history
    
    def get_game_summary(self) -> str:
        """Generate a summary of the game."""
        summary = f"Game between {self.player_black.name} (B) and {self.player_white.name} (W)\n\n"
        
        # Count pieces
        black_count = sum(row.count('B') for row in self.board)
        white_count = sum(row.count('W') for row in self.board)
        
        # Add move history
        summary += "Move history:\n"
        for i, move in enumerate(self.move_history):
            if move[1] is None:  # Skip
                summary += f"{i+1}. {move[0]}: skipped turn\n"
            else:
                summary += f"{i+1}. {move[0]}: [{move[1]}, {move[2]}] (flipped {move[3]} pieces)\n"
        
        # Add result
        if self.game_over:
            if self.winner:
                summary += f"\nWinner: {self.winner.name} ({self.winner.color})"
            else:
                summary += "\nResult: Draw"
            summary += f"\nFinal Score: Black {black_count} - {white_count} White"
        else:
            summary += f"\nGame in progress. {self.current_player.name}'s turn ({self.current_player.color})"
            summary += f"\nCurrent Score: Black {black_count} - {white_count} White"
        
        return summary


# Define available LLM players
AVAILABLE_PLAYERS = {
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "model": "gemini-2.0-flash",
        "api_key": GEMINI_API_KEY,
        "provider": "gemini",
        "type": "llm"
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "model": "gemini-1.5-flash",
        "api_key": GEMINI_API_KEY,
        "provider": "gemini",
        "type": "llm"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o-mini",
        "model": "gpt-4o-mini",
        "api_key": OPENAI_API_KEY,
        "provider": "openai",
        "type": "llm"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "model": "gpt-4o",
        "api_key": OPENAI_API_KEY,
        "provider": "openai",
        "type": "llm"
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5-Turbo",
        "model": "gpt-3.5-turbo",
        "api_key": OPENAI_API_KEY,
        "provider": "openai",
        "type": "llm"
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "model": "claude-3-5-sonnet-20240620",
        "api_key": ANTHROPIC_API_KEY,
        "provider": "anthropic",
        "type": "llm"
    },
    "claude-3.7-sonnet": {
        "name": "Claude 3.7 Sonnet",
        "model": "claude-3-7-sonnet-20250219",
        "api_key": ANTHROPIC_API_KEY,
        "provider": "anthropic",
        "type": "llm"
    },
    "human": {
        "name": "Human Player",
        "type": "human"
    }
}


def create_player(player_id: str, color: str, custom_name: Optional[str] = None) -> Player:
    """
    Create a player instance based on player_id.
    
    Args:
        player_id: ID of the player from AVAILABLE_PLAYERS or "human"
        color: 'B' for black or 'W' for white
        custom_name: Optional custom name for the player
    
    Returns:
        A Player instance (either LLMPlayer or HumanPlayer)
    """
    if player_id not in AVAILABLE_PLAYERS:
        raise ValueError(f"Player '{player_id}' not found in available players")
    
    player_config = AVAILABLE_PLAYERS[player_id]
    player_name = custom_name or player_config["name"]
    
    if player_config["type"] == "human":
        return HumanPlayer(name=player_name, color=color)
    else:
        return LLMPlayer(
            name=player_name,
            color=color,
            model=player_config["model"],
            api_key=player_config["api_key"],
            provider=player_config["provider"]
        )


def create_game(player1_id: str, player2_id: str, 
                player1_name: Optional[str] = None, 
                player2_name: Optional[str] = None,
                delay_seconds: float = 1.0) -> ReversiGame:
    """
    Create a new game between two players.
    
    Args:
        player1_id: ID of the first player (Black) from AVAILABLE_PLAYERS
        player2_id: ID of the second player (White) from AVAILABLE_PLAYERS
        player1_name: Optional custom name for player 1
        player2_name: Optional custom name for player 2
        delay_seconds: Delay between moves for better visualization
    
    Returns:
        A ReversiGame instance
    """
    # Create player Black (goes first)
    player1 = create_player(player1_id, "B", player1_name)
    
    # Create player White
    player2 = create_player(player2_id, "W", player2_name)
    
    # Create and return the game
    return ReversiGame(player1, player2, delay_seconds=delay_seconds)


def play_complete_game(player1_id: str, player2_id: str, 
                       player1_name: Optional[str] = None, 
                       player2_name: Optional[str] = None,
                       delay_seconds: float = 1.0,
                       verbose: bool = True,
                       use_display: bool = False) -> Dict[str, Any]:
    """
    Play a complete game between two players from start to finish.
    
    Args:
        player1_id: ID of the first player (Black) from AVAILABLE_PLAYERS
        player2_id: ID of the second player (White) from AVAILABLE_PLAYERS
        player1_name: Optional custom name for player 1
        player2_name: Optional custom name for player 2
        delay_seconds: Delay between moves for better visualization
        verbose: Whether to print game progress
        use_display: Whether to use graphical display (for notebooks)
    
    Returns:
        A dictionary with game results
    """
    # Create the game
    game = create_game(player1_id, player2_id, player1_name, player2_name, delay_seconds)
    
    # Play the game
    winner, moves = game.play_game(verbose=verbose, use_display=use_display)
    
    # Count final pieces
    black_count = sum(row.count('B') for row in game.board)
    white_count = sum(row.count('W') for row in game.board)
    
    # Print summary
    if verbose:
        print("\n" + "="*50)
        print(game.get_game_summary())
        print("="*50)
    
    # Return results
    return {
        "winner": winner.name if winner else "Draw",
        "winner_color": winner.color if winner else None,
        "score": {"B": black_count, "W": white_count},
        "moves": moves,
        "summary": game.get_game_summary(),
        "final_board": game.get_board_state()
    }


# Import sys for notebook detection
import sys

# Example notebook usage:
"""
# Create a game with a human player against Claude
game = create_game("human", "claude-3.5-sonnet", player1_name="You")

# View the initial board
game.display_board()

# Make moves one at a time
result = game.next_move()
game.display_board()

# Continue until game over
result = game.next_move()
game.display_board()

# Get a summary of the game so far
print(game.get_game_summary())
"""

# Example of playing a complete game in a notebook:
"""
# Play a complete game between Gemini and GPT
results = play_complete_game(
    "gemini-2.0-flash", "gpt-4o-mini",
    delay_seconds=0.5,
    verbose=True,
    use_display=True  # Use graphical display
)

# View the results
print(f"Winner: {results['winner']}")
print(f"Score: Black {results['score']['B']} - {results['score']['W']} White")
"""

# Example usage in a regular Python script
if __name__ == "__main__" and 'ipykernel' not in sys.modules:
    print("REVERSI GAME (4x4)")
    print("==================")
    print("Available players:")
    for player_id, config in AVAILABLE_PLAYERS.items():
        print(f"- {player_id}: {config['name']}")
    print()
    
    # Get player selections
    player1_id = input("Select player 1 (Black) [default: human]: ").strip() or "human"
    player1_name = input("Player 1 name [default: use standard name]: ").strip() or None
    
    player2_id = input("Select player 2 (White) [default: gemini-2.0-flash]: ").strip() or "gemini-2.0-flash"
    player2_name = input("Player 2 name [default: use standard name]: ").strip() or None
    
    # Simple validation
    if player1_id not in AVAILABLE_PLAYERS:
        print(f"Invalid player 1 selection. Using default (human).")
        player1_id = "human"
    
    if player2_id not in AVAILABLE_PLAYERS:
        print(f"Invalid player 2 selection. Using default (gemini-2.0-flash).")
        player2_id = "gemini-2.0-flash"
    
    # Check if human players are involved
    has_human_player = (player1_id == "human" or player2_id == "human" or
                        AVAILABLE_PLAYERS[player1_id]["type"] == "human" or 
                        AVAILABLE_PLAYERS[player2_id]["type"] == "human")
    
    # Ask for mode with appropriate guidance
    if has_human_player:
        print("\nNote: Human player detected. Only interactive mode is available.")
        mode = "interactive"
    else:
        mode = input("Select mode (interactive/complete) [default: interactive]: ").strip().lower() or "interactive"
    
    if mode == "interactive" or has_human_player:
        # Interactive mode
        if mode == "complete" and has_human_player:
            print("\nSwitching to interactive mode because human players are involved.")
        
        game = create_game(player1_id, player2_id, player1_name, player2_name)
        print(f"\nStarting an interactive game: {game.player_black.name} (B) vs {game.player_white.name} (W)")
        game.print_board()
        
        # Main game loop
        while not game.game_over:
            # Execute next move
            result = game.next_move()
            
            # Check if game is over
            if result["status"] == "game_over":
                print("\n" + "="*50)
                print(game.get_game_summary())
                print("="*50)
                break
    
    else:
        # Play complete game
        results = play_complete_game(
            player1_id, player2_id, 
            player1_name, player2_name,
            delay_seconds=1.0,
            verbose=True,
            use_display=False
        )