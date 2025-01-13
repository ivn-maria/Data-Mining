import math


def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)


def check_winner(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != " ":
            return row[0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != " ":
            return board[0][col]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != " ":
        return board[0][2]
    return None


def is_full(board):
    return all(cell != " " for row in board for cell in row)


# depth is the current depth of the recursion, used to choose quicker wins or delayed losses
# alpha is the best score that the maximizing player is guaranteed
# beta is the best score that the minimizing player is guaranteed
def minimax(board, alpha, beta, is_maximizing, computer_symbol):
    if computer_symbol == "0":
        player_symbol = "X"
    else:
        player_symbol = "O"

    winner = check_winner(board)
    if winner == computer_symbol:
        # higher score, quick win preferred
        return 10
    if winner == player_symbol:
        # lower score, slower loss preferred
        return -10
    if is_full(board):
        return 0

    if is_maximizing:
        # maximizing player (the computer) aims to get the highest possible score
        max_eval = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = computer_symbol
                    evaluation = minimax(board, alpha, beta, False, computer_symbol)
                    board[i][j] = " "
                    max_eval = max(max_eval, evaluation)
                    alpha = max(alpha, evaluation)
                    # alpha-beta pruning to avoid unnecessary calculations
                    # if the score at a node is worse than the current best-known option, it prunes the search
                    if beta <= alpha:
                        break
        return max_eval
    else:
        # minimizing player (the player) aims to get the lowest possible score
        min_eval = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = player_symbol
                    evaluation = minimax(board, alpha, beta, True, computer_symbol)
                    board[i][j] = " "
                    min_eval = min(min_eval, evaluation)
                    beta = min(beta, evaluation)
                    # alpha-beta pruning to avoid unnecessary calculations
                    # if the score at a node is worse than the current best-known option, it prunes the search
                    if beta <= alpha:
                        break
        return min_eval


# finds the best move for the maximizing player
# tracks the best move for the computer (the one with the highest score) and returns it
def find_best_move(board, computer_symbol):
    if computer_symbol == "O":
        player_symbol = "X"
    else:
        player_symbol = "O"

    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = player_symbol
                board[i][j] = " "
                if check_winner(board) == player_symbol:
                    best_move = (i, j)

    best_value = -math.inf
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = computer_symbol
                move_value = minimax(board, -math.inf, math.inf, False, computer_symbol)
                board[i][j] = " "
                if move_value > best_value:
                    best_value = move_value
                    best_move = (i, j)
    return best_move


def players_move(board, player_symbol):
    while True:
        try:
            row, col = map(int, input("Your move (row and column 1 to 3, separated by a space): ").split())
            row, col = row - 1, col - 1
            if board[row][col] == " ":
                board[row][col] = player_symbol
                break
            else:
                print("The field is already taken. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    print_board(board)

    if check_winner(board):
        print(f"Player wins!")
        return 1
    if is_full(board):
        print("The game ended in a draw!")
        return 1


def computers_move(board, computer_symbol):
    move = find_best_move(board, computer_symbol)
    if move:
        board[move[0]][move[1]] = computer_symbol
    print("Computer's move:")
    print_board(board)

    if check_winner(board):
        print(f"Computer wins!")
        return 1
    if is_full(board):
        print("The game ended in a draw!")
        return 1


def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    order = input("Do you want to start first? (yes/no): ").strip().lower()
    if order != "yes" and order != "no":
        print("Invalid input!")
        exit()
    player_first = order == "yes"
    player_symbol, computer_symbol = ("O", "X") if player_first else ("X", "O")

    while True:
        if player_first:
            if players_move(board, player_symbol) == 1:
                break
            if computers_move(board, computer_symbol) == 1:
                break
        else:
            if computers_move(board, computer_symbol) == 1:
                break
            if players_move(board, player_symbol) == 1:
                break


if __name__ == "__main__":
    play_game()
