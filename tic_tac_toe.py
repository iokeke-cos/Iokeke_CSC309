import math

X = "X"  # AI
O = "O"  # Human
EMPTY = " "

# Node counters
minimax_nodes = 0
alphabeta_nodes = 0


def print_board(board):
    print()
    for i in range(0, 9, 3):
        print(f" {board[i]} | {board[i+1]} | {board[i+2]} ")
        if i < 6:
            print("---+---+---")
    print()


def print_reference_board():
    print("Board positions:\n")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")
    print()


def check_winner(board):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for a, b, c in win_patterns:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board):
    return all(cell != EMPTY for cell in board)


# ---------------- MINIMAX (for comparison) ----------------
def minimax(board, is_maximizing):
    global minimax_nodes
    minimax_nodes += 1

    winner = check_winner(board)

    if winner == X:
        return 1
    if winner == O:
        return -1
    if is_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = X
                score = minimax(board, False)
                board[i] = EMPTY
                best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = O
                score = minimax(board, True)
                board[i] = EMPTY
                best_score = min(best_score, score)
        return best_score


# ---------------- ALPHA-BETA PRUNING ----------------
def alpha_beta(board, alpha, beta, is_maximizing):
    global alphabeta_nodes
    alphabeta_nodes += 1

    winner = check_winner(board)

    if winner == X:
        return 1
    if winner == O:
        return -1
    if is_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = X
                score = alpha_beta(board, alpha, beta, False)
                board[i] = EMPTY

                best_score = max(best_score, score)
                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break  # PRUNE

        return best_score

    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = O
                score = alpha_beta(board, alpha, beta, True)
                board[i] = EMPTY

                best_score = min(best_score, score)
                beta = min(beta, best_score)

                if beta <= alpha:
                    break  # PRUNE

        return best_score


# ---------------- AI MOVE ----------------
def best_ai_move(board):
    best_score = -math.inf
    move = None

    for i in range(9):
        if board[i] == EMPTY:
            board[i] = X
            score = alpha_beta(board, -math.inf, math.inf, False)
            board[i] = EMPTY

            if score > best_score:
                best_score = score
                move = i

    return move


def ai_move(board):
    move = best_ai_move(board)
    board[move] = X
    print(f"AI chooses position {move}")


# ---------------- HUMAN MOVE ----------------
def human_move(board):
    while True:
        try:
            choice = int(input("Enter your move (0-8): "))
            if choice < 0 or choice > 8:
                print("Enter a number from 0 to 8.")
                continue

            if board[choice] != EMPTY:
                print("Position already taken.")
                continue

            board[choice] = O
            break

        except ValueError:
            print("Invalid input.")


# ---------------- GAME LOOP ----------------
def play_game():
    board = [EMPTY] * 9

    print("Welcome to Tic-Tac-Toe")
    print("You are O, AI is X")
    print_reference_board()

    current_player = O

    while True:
        print_board(board)

        winner = check_winner(board)
        if winner == X:
            print("AI wins.")
            break
        elif winner == O:
            print("You win.")
            break
        elif is_full(board):
            print("It's a draw.")
            break

        if current_player == O:
            human_move(board)
            current_player = X
        else:
            ai_move(board)
            current_player = O


# ---------------- NODE COMPARISON ----------------
def compare_algorithms():
    global minimax_nodes, alphabeta_nodes

    test_board = [EMPTY] * 9

    minimax_nodes = 0
    alphabeta_nodes = 0

    minimax(test_board, True)
    alpha_beta(test_board, -math.inf, math.inf, True)

    print("\n--- Node Comparison ---")
    print("Minimax nodes:", minimax_nodes)
    print("Alpha-Beta nodes:", alphabeta_nodes)
    print("Nodes pruned:", minimax_nodes - alphabeta_nodes)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    compare_algorithms()
    play_game()