import numpy as np
import random
import matplotlib.pyplot as plt

class TicTacToeAI:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        # The Q-Table: Maps (Board State) -> [Value of each Move 0-8]
        # Using a dictionary because there are too many states for a simple array
        self.q_table = {} 
        self.epsilon = epsilon # Exploration rate (Try random moves?)
        self.alpha = alpha     # Learning rate (How fast to accept new info?)
        self.gamma = gamma     # Discount factor (Care about future wins?)

    def get_state_key(self, board):
        # Convert board array to a string tuple for dictionary key
        return tuple(board)

    def choose_action(self, board):
        state = self.get_state_key(board)
        available_moves = [i for i, x in enumerate(board) if x == 0]

        # Explore: Random move
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)

        # Exploit: Choose best known move
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        
        # Get Q-values for all moves, pick the best LEGAL move
        q_values = self.q_table[state]
        # We set illegal moves to -infinity so it never picks them
        legal_q_values = [q_values[i] if i in available_moves else -np.inf for i in range(9)]
        
        # Randomly choose among the best moves (if tie)
        max_val = np.max(legal_q_values)
        best_moves = [i for i, v in enumerate(legal_q_values) if v == max_val]
        return random.choice(best_moves)

    def learn(self, state, action, reward, next_state, game_over):
        # Q-Learning Formula
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(9)

        current_q = self.q_table[state_key][action]
        
        # If game over, there is no "future max q"
        max_future_q = np.max(self.q_table[next_state_key]) if not game_over else 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_key][action] = new_q

# --- GAME ENGINE ---

def check_winner(board, player):
    # Winning combinations
    wins = [[0,1,2],[3,4,5],[6,7,8], # Horizontal
            [0,3,6],[1,4,7],[2,5,8], # Vertical
            [0,4,8],[2,4,6]]         # Diagonal
    
    for w in wins:
        if all(board[i] == player for i in w):
            return True
    return False

def train(iterations=10000):
    ai = TicTacToeAI()
    print(f"Training AI for {iterations} games...")
    
    history_rewards = []
    
    for i in range(iterations):
        board = np.zeros(9, dtype=int)
        game_over = False
        # AI plays as 1, Opponent (Random) plays as -1
        turn = 1 
        
        # Keep track of moves for learning
        ai_last_state = None
        ai_last_action = None
        
        while not game_over:
            if turn == 1: # AI Turn
                action = ai.choose_action(board)
                ai_last_state = board.copy()
                ai_last_action = action
                
                board[action] = 1
                
                if check_winner(board, 1):
                    ai.learn(ai_last_state, action, 10, board, True) # Big reward for win
                    game_over = True
                    history_rewards.append(1)
                elif 0 not in board: # Draw
                    ai.learn(ai_last_state, action, 2, board, True) # Small reward for draw
                    game_over = True
                    history_rewards.append(0)
                else:
                    turn = -1
                    
            else: # Opponent Turn (Random Player)
                # Note: To make AI smarter, you can make this opponent smarter too
                available = [i for i, x in enumerate(board) if x == 0]
                action = random.choice(available)
                board[action] = -1
                
                if check_winner(board, -1):
                    # AI lost! Punish the LAST move it made
                    ai.learn(ai_last_state, ai_last_action, -10, board, True)
                    game_over = True
                    history_rewards.append(-1)
                elif 0 not in board: # Draw
                    ai.learn(ai_last_state, ai_last_action, 2, board, True)
                    game_over = True
                    history_rewards.append(0)
                else:
                    # Game continues, AI learns a small "living reward" (0)
                    ai.learn(ai_last_state, ai_last_action, 0, board, False)
                    turn = 1
                    
    print("Training Complete.")
    return ai, history_rewards

# --- PLAY AGAINST IT ---

def play_game(ai):
    board = np.zeros(9, dtype=int)
    print("\n--- HUMAN vs AI ---")
    print("You are X (-1). AI is O (1).")
    print("Board positions:")
    print("0 1 2\n3 4 5\n6 7 8\n")
    
    ai.epsilon = 0 # Turn off randomness (Play to win)
    
    while True:
        # AI Turn
        action = ai.choose_action(board)
        board[action] = 1
        print(f"AI chose position {action}")
        
        # Render Board
        symbols = {0: '.', 1: 'O', -1: 'X'}
        print(f"\n {symbols[board[0]]} {symbols[board[1]]} {symbols[board[2]]}")
        print(f" {symbols[board[3]]} {symbols[board[4]]} {symbols[board[5]]}")
        print(f" {symbols[board[6]]} {symbols[board[7]]} {symbols[board[8]]}\n")
        
        if check_winner(board, 1):
            print("AI WINS! (As expected, mere mortal)")
            break
        if 0 not in board:
            print("DRAW!")
            break
            
        # Human Turn
        while True:
            try:
                move = int(input("Your move (0-8): "))
                if board[move] == 0:
                    board[move] = -1
                    break
                else:
                    print("Occupied!")
            except:
                print("Invalid input.")
                
        if check_winner(board, -1):
            print("YOU WIN! (Impossible...)")
            break
        if 0 not in board:
            print("DRAW!")
            break

if __name__ == "__main__":
    ai_brain, history = train(50000)
    
    # Plot training progress
    # Moving average of wins/losses
    def moving_average(a, n=1000):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(history))
    plt.title("AI Performance (Moving Average)")
    plt.xlabel("Games Played")
    plt.ylabel("Score (Win=1, Loss=-1)")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()
    
    # Start Interactive Game
    play_game(ai_brain)