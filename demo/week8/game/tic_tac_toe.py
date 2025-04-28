import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# API keys configuration - set your keys here directly
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"

# Abstract base class for all players (LLMs and humans)
class Player:
    def __init__(self, name: str, marker: str):
        self.name = name
        self.marker = marker  # 'X' or 'O'
    
    def get_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Get the next move from the player."""
        raise NotImplementedError("Subclasses must implement this method")


# Human player that takes input from the terminal
class HumanPlayer(Player):
    def __init__(self, name: str, marker: str):
        super().__init__(name, marker)
    
    def get_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Get the next move from the human player via terminal input."""
        print(f"\n{self.name}'s turn ({self.marker})")
        self._print_board(board)
        
        while True:
            try:
                move_input = input(f"Enter your move as 'row,col' (0-2): ")
                
                # Parse the input
                parts = move_input.strip().split(',')
                if len(parts) != 2:
                    raise ValueError("Input must be in the format 'row,col'")
                
                row, col = int(parts[0]), int(parts[1])
                
                # Validate the move
                if not (0 <= row <= 2 and 0 <= col <= 2):
                    print("Invalid move: Position must be between 0 and 2")
                    continue
                
                if board[row][col] != '_':
                    print("Invalid move: Position already occupied")
                    continue
                
                return row, col
                
            except ValueError as e:
                print(f"Invalid input: {str(e)}. Try again.")
    
    def _print_board(self, board: List[List[str]]):
        """Print the current state of the board."""
        print("\nCurrent board:")
        print("  0 1 2")
        for i, row in enumerate(board):
            print(f"{i} {' '.join(row)}")
        print()


# LLM player class that handles multiple LLM providers
class LLMPlayer(Player):
    def __init__(self, 
                 name: str, 
                 marker: str, 
                 model: str, 
                 api_key: str, 
                 provider: Literal["gemini", "openai", "anthropic"] = "gemini"):
        super().__init__(name, marker)
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
        You are playing a game of Tic Tac Toe. The board is represented as a 3x3 grid. Empty spaces are represented 
        by '_', your moves are represented by '{marker}'. Your opponent's moves are represented by 
        '{opponent_marker}'. You are playing as {marker}.
        
        The board positions are as follows:
        [0,0] [0,1] [0,2]
        [1,0] [1,1] [1,2]
        [2,0] [2,1] [2,2]
        
        When it's your turn, analyze the board and provide your next move in the format: [row, column]
        For example, if you want to place your marker in the top-right corner, respond with: [0, 2]
        
        Choose your moves strategically to either win the game or block your opponent from winning.
        Respond ONLY with your move coordinates in the format [row, column] - no explanation or other text.
        """.format(marker=self.marker, opponent_marker='O' if self.marker == 'X' else 'X')
        
        # For OpenAI and Anthropic, we can use the system role
        if self.provider in ["openai", "anthropic"]:
            self.update_history("system", self.system_prompt)
        else:
            # For Gemini, we'll add the system prompt as a user message since it doesn't support system role
            first_prompt = f"Instructions for playing Tic Tac Toe:\n{self.system_prompt}\nDo you understand these instructions? Remember to only respond with move coordinates in the format [row, column]."
            self.update_history("user", first_prompt)
            self.update_history("assistant", "I understand. I'll respond with only the move coordinates in the format [row, column].")
    
    def update_history(self, role: str, content: str):
        """Add a message to the chat history."""
        # For Gemini, we skip system messages since they're not supported
        if self.provider == "gemini" and role == "system":
            return
            
        self.chat_history.append({"role": role, "content": content})
    
    def _format_board_for_prompt(self, board: List[List[str]]) -> str:
        """Format the board as a string for the prompt."""
        board_str = "Current board:\n"
        for row in board:
            board_str += " ".join(row) + "\n"
        return board_str
    
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
    
    def get_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Get the next move from the LLM model."""
        board_str = self._format_board_for_prompt(board)
        prompt = f"{board_str}\nIt's your turn. Where would you like to place your {self.marker}? Respond only with [row, column]."
        
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
                return self._get_random_valid_move(board)
            
            response_data = response.json()
            response_text = self._parse_response(response_data)
            
            # Add the response to chat history
            self.update_history("assistant", response_text)
            
            # Parse the move from the response
            import re
            match = re.search(r'\[(\d+)\s*,\s*(\d+)\]', response_text)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == '_':
                    return row, col
            
            # If we couldn't parse a valid move, fall back to random
            print(f"Could not parse valid move from: {response_text}")
            return self._get_random_valid_move(board)
            
        except Exception as e:
            print(f"Error with {self.provider} API request: {str(e)}")
            return self._get_random_valid_move(board)
    
    def _get_random_valid_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Get a random valid move as a fallback."""
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if board[r][c] == '_':
                    valid_moves.append((r, c))
        
        if not valid_moves:
            raise ValueError("No valid moves available (board is full)")
        
        return random.choice(valid_moves)


class TicTacToeGame:
    def __init__(self, player_x: Player, player_o: Player, delay_seconds: float = 1.0):
        self.board = [['_' for _ in range(3)] for _ in range(3)]
        self.player_x = player_x  # X goes first
        self.player_o = player_o
        self.current_player = player_x
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.delay_seconds = delay_seconds
    
    def get_board_state(self):
        """Return a copy of the current board state."""
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
        """Print the current state of the board as text."""
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
        """Attempt to make a move at the specified position."""
        if self.game_over:
            print("Game is already over!")
            return False
        
        if not (0 <= row <= 2 and 0 <= col <= 2):
            print(f"Invalid move: [{row}, {col}] is out of bounds")
            return False
        
        if self.board[row][col] != '_':
            print(f"Invalid move: [{row}, {col}] is already occupied")
            return False
        
        self.board[row][col] = self.current_player.marker
        self.move_history.append((self.current_player.name, row, col))
        
        return True
    
    def check_winner(self) -> Optional[str]:
        """Check if there is a winner or if the game is a draw."""
        # Check rows
        for row in self.board:
            if row[0] != '_' and row[0] == row[1] == row[2]:
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] != '_' and self.board[0][col] == self.board[1][col] == self.board[2][col]:
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] != '_' and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        
        if self.board[0][2] != '_' and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]
        
        # Check if board is full (draw)
        is_full = all(cell != '_' for row in self.board for cell in row)
        if is_full:
            return "Draw"
        
        return None
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player = self.player_o if self.current_player == self.player_x else self.player_x
    
    def next_move(self) -> Dict[str, Any]:
        """Execute the next move and return the result (for interactive mode)."""
        if self.game_over:
            return {
                "status": "game_over",
                "winner": self.winner.name if self.winner else "Draw",
                "board": self.get_board_state(),
                "move_history": self.move_history
            }
        
        # Get move from current player
        player_name = self.current_player.name
        player_marker = self.current_player.marker
        
        print(f"{player_name}'s turn ({player_marker})...")
        
        # Get and execute the move
        row, col = self.current_player.get_move(self.board)
        print(f"{player_name} chooses: [{row}, {col}]")
        
        # Make the move
        if self.make_move(row, col):
            # Check for a winner
            result = self.check_winner()
            if result:
                self.game_over = True
                if result == "Draw":
                    self.winner = None
                    print("Game over! It's a draw!")
                else:
                    self.winner = self.current_player
                    print(f"Game over! {self.winner.name} ({result}) wins!")
            else:
                # Switch to the other player
                self.switch_player()
        else:
            # Invalid move, try again with random move
            print(f"Invalid move from {self.current_player.name}. Using random valid move instead...")
            valid_moves = []
            for r in range(3):
                for c in range(3):
                    if self.board[r][c] == '_':
                        valid_moves.append((r, c))
            
            if valid_moves:
                row, col = random.choice(valid_moves)
                self.make_move(row, col)
                
                # Check for a winner after random move
                result = self.check_winner()
                if result:
                    self.game_over = True
                    if result == "Draw":
                        self.winner = None
                        print("Game over! It's a draw!")
                    else:
                        self.winner = self.current_player
                        print(f"Game over! {self.winner.name} ({result}) wins!")
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
    
    def play_game(self, verbose: bool = True, use_display: bool = False) -> Tuple[Optional[Player], List[Tuple[str, int, int]]]:
        """Play a full game of Tic Tac Toe (for complete mode)."""
        if verbose:
            print(f"Starting a new game: {self.player_x.name} (X) vs {self.player_o.name} (O)")
            if use_display:
                self.display_board()
            else:
                self.print_board()
        
        while not self.game_over:
            # Get move from current player
            player_name = self.current_player.name
            player_marker = self.current_player.marker
            
            if verbose:
                print(f"\n{player_name}'s turn ({player_marker})...")
            
            # Get and execute the move
            row, col = self.current_player.get_move(self.board)
            
            if verbose:
                print(f"{player_name} chooses: [{row}, {col}]")
            
            # Make the move
            if self.make_move(row, col):
                if verbose:
                    if use_display:
                        if 'ipykernel' in sys.modules:  # Check if running in notebook
                            clear_output(wait=True)  # Clear previous board
                        self.display_board()
                    else:
                        self.print_board()
                
                # Check for a winner
                result = self.check_winner()
                if result:
                    self.game_over = True
                    if result == "Draw":
                        self.winner = None
                        if verbose:
                            print("Game over! It's a draw!")
                    else:
                        self.winner = self.current_player
                        if verbose:
                            print(f"Game over! {self.winner.name} ({result}) wins!")
                else:
                    # Switch to the other player
                    self.switch_player()
                    
                    # Add a small delay between moves for better visualization
                    if self.delay_seconds > 0 and verbose and not isinstance(self.current_player, HumanPlayer):
                        time.sleep(self.delay_seconds)
            else:
                # Invalid move, try again with random move (if not human)
                if verbose:
                    print(f"Invalid move from {self.current_player.name}. Trying again...")
                
                # If it's an LLM player, use a random valid move
                if not isinstance(self.current_player, HumanPlayer):
                    valid_moves = []
                    for r in range(3):
                        for c in range(3):
                            if self.board[r][c] == '_':
                                valid_moves.append((r, c))
                    
                    if valid_moves:
                        row, col = random.choice(valid_moves)
                        self.make_move(row, col)
                        
                        if verbose:
                            print(f"Using random move: [{row}, {col}]")
                            if use_display:
                                self.display_board()
                            else:
                                self.print_board()
                        
                        # Check for a winner after random move
                        result = self.check_winner()
                        if result:
                            self.game_over = True
                            if result == "Draw":
                                self.winner = None
                                if verbose:
                                    print("Game over! It's a draw!")
                            else:
                                self.winner = self.current_player
                                if verbose:
                                    print(f"Game over! {self.winner.name} ({result}) wins!")
                        else:
                            # Switch to the other player
                            self.switch_player()
        
        return self.winner, self.move_history
    
    def get_game_summary(self) -> str:
        """Generate a summary of the game."""
        summary = f"Game between {self.player_x.name} (X) and {self.player_o.name} (O)\n\n"
        
        # Add move history
        summary += "Move history:\n"
        for i, (player_name, row, col) in enumerate(self.move_history):
            summary += f"{i+1}. {player_name}: [{row}, {col}]\n"
        
        # Add result
        if self.game_over:
            if self.winner:
                summary += f"\nWinner: {self.winner.name} ({self.winner.marker})"
            else:
                summary += "\nResult: Draw"
        else:
            summary += f"\nGame in progress. {self.current_player.name}'s turn ({self.current_player.marker})"
        
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


def create_player(player_id: str, marker: str, custom_name: Optional[str] = None) -> Player:
    """
    Create a player instance based on player_id.
    
    Args:
        player_id: ID of the player from AVAILABLE_PLAYERS or "human"
        marker: 'X' or 'O'
        custom_name: Optional custom name for the player
    
    Returns:
        A Player instance (either LLMPlayer or HumanPlayer)
    """
    if player_id not in AVAILABLE_PLAYERS:
        raise ValueError(f"Player '{player_id}' not found in available players")
    
    player_config = AVAILABLE_PLAYERS[player_id]
    player_name = custom_name or player_config["name"]
    
    if player_config["type"] == "human":
        return HumanPlayer(name=player_name, marker=marker)
    else:
        return LLMPlayer(
            name=player_name,
            marker=marker,
            model=player_config["model"],
            api_key=player_config["api_key"],
            provider=player_config["provider"]
        )


def create_game(player1_id: str, player2_id: str, 
                player1_name: Optional[str] = None, 
                player2_name: Optional[str] = None,
                delay_seconds: float = 1.0) -> TicTacToeGame:
    """
    Create a new game between two players.
    
    Args:
        player1_id: ID of the first player (X) from AVAILABLE_PLAYERS
        player2_id: ID of the second player (O) from AVAILABLE_PLAYERS
        player1_name: Optional custom name for player 1
        player2_name: Optional custom name for player 2
        delay_seconds: Delay between moves for better visualization
    
    Returns:
        A TicTacToeGame instance
    """
    # Create player X (goes first)
    player1 = create_player(player1_id, "X", player1_name)
    
    # Create player O
    player2 = create_player(player2_id, "O", player2_name)
    
    # Create and return the game
    return TicTacToeGame(player1, player2, delay_seconds=delay_seconds)


def play_complete_game(player1_id: str, player2_id: str, 
                       player1_name: Optional[str] = None, 
                       player2_name: Optional[str] = None,
                       delay_seconds: float = 1.0,
                       verbose: bool = True,
                       use_display: bool = False) -> Dict[str, Any]:
    """
    Play a complete game between two players from start to finish.
    
    Args:
        player1_id: ID of the first player (X) from AVAILABLE_PLAYERS
        player2_id: ID of the second player (O) from AVAILABLE_PLAYERS
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
    
    # Print summary
    if verbose:
        print("\n" + "="*50)
        print(game.get_game_summary())
        print("="*50)
    
    # Return results
    return {
        "winner": winner.name if winner else "Draw",
        "winner_marker": winner.marker if winner else None,
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
print(f"Number of moves: {len(results['moves'])}")
"""

# Example usage in a regular Python script
if __name__ == "__main__" and 'ipykernel' not in sys.modules:
    print("TIC TAC TOE GAME")
    print("================")
    print("Available players:")
    for player_id, config in AVAILABLE_PLAYERS.items():
        print(f"- {player_id}: {config['name']}")
    print()
    
    # Get player selections
    player1_id = input("Select player 1 (X) [default: human]: ").strip() or "human"
    player1_name = input("Player 1 name [default: use standard name]: ").strip() or None
    
    player2_id = input("Select player 2 (O) [default: gemini-2.0-flash]: ").strip() or "gemini-2.0-flash"
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
        print(f"\nStarting an interactive game: {game.player_x.name} (X) vs {game.player_o.name} (O)")
        game.print_board()
        
        # Main game loop
        while not game.game_over:
            # Execute next move
            result = game.next_move()
            game.print_board()
            
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