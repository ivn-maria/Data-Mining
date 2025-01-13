import time
from heapq import heappush, heappop


def print_board(state):
    result = []

    for x in state:
        if x == 1:
            result.append('>')
        elif x == -1:
            result.append('<')
        else:
            result.append('_')

    output = ''.join(result)

    print(output)


def frog_puzzle(n):
    # 1 is a frog turned to the right
    # 0 is the empty space
    # -1 is a frog turned to the left
    initial_state = tuple([1] * n + [0] + [-1] * n)
    target_state = tuple([-1] * n + [0] + [1] * n)

    # stores:
    # score which is a heuristic score representing the priority
    # state which is the current state of the board
    # empty_index which is the index of the empty space
    # path which is a list of states that led to the current state
    priority_queue = []
    heappush(priority_queue, (0, initial_state, initial_state.index(0), [initial_state]))

    # stores states that have been visited
    visited = set()
    visited.add(initial_state)

    # (-1, 1) is the move left 1 step
    # (-1, 2) is the jump left over 1 frog
    # (1, 1) is the move right 1 step
    # (1, 2) is the jump right over 1 frog
    moves = [(-1, 1), (-1, 2), (1, 1), (1, 2)]

    while priority_queue:
        score, state, empty_index, path = heappop(priority_queue)

        # checks if the puzzle is solved
        if state == target_state:
            for move in path:
                print_board(move)
            return True

        for direction, jump in moves:

            # calculate the new position of the empty space
            new_empty = empty_index + direction * jump

            #  validate that the new position is within bounds and that the new position is not 0
            if len(state) > new_empty >= 0 != state[new_empty]:

                # swap the empty space with the frog at the new position
                new_state = list(state)
                new_state[empty_index], new_state[new_empty] = new_state[new_empty], 0
                new_state_tuple = tuple(new_state)

                if new_state_tuple not in visited:

                    # add the new state to the visited set if it has not been visited before
                    visited.add(new_state_tuple)

                    # the score is 2 for a jump move and 1 for a normal move
                    new_score = (2 if jump == 2 else 1)

                    # push the new state into the priority queue with its score, empty space index, and updated path
                    heappush(priority_queue, (new_score, new_state_tuple, new_empty, path + [new_state_tuple]))

    return False


N = int(input('N: '))
start_time = time.time()
frog_puzzle(N)
end_time = time.time()
execution_time = round(end_time - start_time, 4)
print(f"Execution time: {execution_time} seconds")
