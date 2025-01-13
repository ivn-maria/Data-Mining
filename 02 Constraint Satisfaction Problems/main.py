import time
import random


def print_board(queens):
    n = len(queens)
    for row in range(n):
        line = []
        for col in range(n):
            if queens[col] == row:
                line.append('*')
            else:
                line.append('_')
        print(' '.join(line))


def min_conflict_column(n, row, row_conflicts, main_diag_conflicts, anti_diag_conflicts):
    min_conflicts_count = float('inf')

    # list of columns with min conflicts
    best_columns = []

    for col in range(n):
        conflicts = (row_conflicts[col] + main_diag_conflicts[row - col + n - 1] + anti_diag_conflicts[row + col])
        if conflicts < min_conflicts_count:
            min_conflicts_count = conflicts
            best_columns = [col]
            if min_conflicts_count == 0:
                return col
        elif conflicts == min_conflicts_count:
            best_columns.append(col)

    return random.choice(best_columns)


def find_conflicted_queens(n, board, row_conflicts, main_diag_conflicts, anti_diag_conflicts):
    # list of rows withs conflicts
    conflicted = []

    for row, col in enumerate(board):
        if (row_conflicts[col] > 1 or
                main_diag_conflicts[row - col + n - 1] > 1 or anti_diag_conflicts[row + col] > 1):
            conflicted.append(row)

    return conflicted


def update_conflicts(row, col, delta, row_conflicts, main_diag_conflicts, anti_diag_conflicts):
    # update conflict counters
    row_conflicts[col] += delta
    main_diag_conflicts[row - col + len(row_conflicts) - 1] += delta
    anti_diag_conflicts[row + col] += delta


def min_conflicts(n):
    max_steps = n * 10

    # initialize board and conflict counters
    board = [-1] * n
    row_conflicts = [0] * n
    main_diag_conflicts = [0] * (2 * n - 1)
    anti_diag_conflicts = [0] * (2 * n - 1)

    # put queens on rows with minimal initial conflicts
    for row in range(n):
        col = min_conflict_column(n, row, row_conflicts, main_diag_conflicts, anti_diag_conflicts)
        board[row] = col
        update_conflicts(row, col, 1, row_conflicts, main_diag_conflicts, anti_diag_conflicts)

    # conflict reduction
    for step in range(max_steps):
        conflicts = find_conflicted_queens(n, board, row_conflicts, main_diag_conflicts, anti_diag_conflicts)

        if not conflicts:
            return board

        # pick a random conflicted queen and move her
        queen_row = random.choice(conflicts)
        old_col = board[queen_row]
        update_conflicts(queen_row, old_col, -1, row_conflicts, main_diag_conflicts, anti_diag_conflicts)
        new_col = min_conflict_column(n, queen_row, row_conflicts, main_diag_conflicts, anti_diag_conflicts)
        board[queen_row] = new_col
        update_conflicts(queen_row, new_col, 1, row_conflicts, main_diag_conflicts, anti_diag_conflicts)

    return board


N = int(input('N: ').strip())

start_time = time.time()
solution = min_conflicts(N)
end_time = time.time()
execution_time = round(end_time - start_time, 4)

if N < 4:
    print("-1")
elif N <= 100:
    print_board(solution)
    print(solution)
print(f"Execution time: {execution_time} seconds")
