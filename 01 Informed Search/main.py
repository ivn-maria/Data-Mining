import sys
import time
import math

# read input
N = int(sys.stdin.readline())
index = int(sys.stdin.readline())
size = int(math.sqrt(N + 1))
board = []
for s in range(size):
    board.extend(map(int, sys.stdin.readline().split()))

# create the goal state
if index == -1:
    index = N
goal_state = list(range(1, N + 1))
goal_state.insert(index, 0)
goal_state = goal_state[:index + 1] + sorted(goal_state[index + 1:])


def is_solvable(state):
    # count the inversions
    inv_count = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] and state[j] and state[i] > state[j]:
                inv_count += 1

    # for odd-sized boards, the number of inversions must be even
    if size % 2 == 1:
        return inv_count % 2 == 0

    # for even-sized boards,the sum the row of the zero and inversions must be even
    else:
        zero_row = state.index(0) // size
        goal_zero_row = goal_state.index(0) // size
        return (inv_count + abs(zero_row - goal_zero_row)) % 2 == 0


# calculate the manhattan/heuristic distance
def manhattan(state):
    distance = 0
    for idx, val in enumerate(state):
        if val != 0:
            goal_idx = goal_state.index(val)
            x1, y1 = divmod(idx, size)
            x2, y2 = divmod(goal_idx, size)
            distance += abs(x1 - x2) + abs(y1 - y2)
    return distance


# determine possible moves from a given zero position:
def get_moves(pos):
    moves = []
    x, y = divmod(pos, size)
    if y > 0:
        moves.append(('right', pos - 1))
    if y < size - 1:
        moves.append(('left', pos + 1))
    if x > 0:
        moves.append(('down', pos - size))
    if x < size - 1:
        moves.append(('up', pos + size))

    return moves


# recursive search function for solving the puzzle using IDA* algorithm
path = []
visited = set()


def search(state, count, distance, zero_pos):
    if state == goal_state:
        return True

    # if estimated distance exceeds the current threshold, return it
    estimated_count = count + manhattan(state)
    if estimated_count > distance:
        return estimated_count

    # track the minimum distance threshold to explore further
    min_distance = float('inf')
    visited.add(tuple(state))
    for move, pos in get_moves(zero_pos):
        # create a new state by swapping the zero with the adjacent tile
        new_state = state[:]
        new_state[zero_pos], new_state[pos] = new_state[pos], new_state[zero_pos]

        # skip the current iteration if the state has been visited
        state_tuple = tuple(new_state)
        if state_tuple in visited:
            continue

        # add the move to the path and recursively search the new state
        path.append(move)
        temp = search(new_state, count + 1, distance, pos)

        if temp is True:
            return True

        # adjust the distance threshold for the next round of the IDA* search
        if temp < min_distance:
            min_distance = temp

        # removes the last move from the path list
        path.pop()

    # allows the state to be revisited in a different path if needed
    visited.remove(tuple(state))

    # if no solution is found, the function returns min_distance as the next threshold
    return min_distance


def ida_search():
    if not is_solvable(board):
        print(-1)
        return

    distance = manhattan(board)
    zero_pos = board.index(0)
    while True:
        temp = search(board, 0, distance, zero_pos)

        if temp is True:
            print(len(path))
            for move in path:
                print(move)
            return

        if temp == float('inf'):
            print(-1)
            return
        distance = temp


start_time = time.time()
ida_search()
end_time = time.time()
execution_time = round(end_time - start_time, 4)
print(f"Execution time: {execution_time} seconds")
