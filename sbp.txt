import sys
import time

original_state = ""
def print_state(filename):
    with open(filename, 'r') as file:
    # Read the entire content of the file into a string
        content = file.read()
        original_state = content
        print(content)

def is_puzzle_solved(filename):
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into cells and convert to integers, then check if -1 is present
            if '-1' in line:
                return False  # Puzzle is not solved if any cell has value -1
    return True  # Puzzle is solved if no cell has value -1



def find_zeros(board):
    zero_positions = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                zero_positions.append((i, j))
    return zero_positions

def possible_replacements(matrix, zero_positions):
    replacements = []

    rows = len(matrix)
    cols = len(matrix[0])

    for zero_position in zero_positions:
        i, j = zero_position

        # Check left
        if j > 0 and matrix[i][j - 1] != 0:
            replacements.append((matrix[i][j - 1], "right"))
        # Check right
        if j < cols - 1 and matrix[i][j + 1] != 0:
            replacements.append((matrix[i][j + 1], "left"))
        # Check up
        if i > 0 and matrix[i - 1][j] != 0:
            replacements.append((matrix[i - 1][j], "down"))
        # Check down
        if i < rows - 1 and matrix[i + 1][j] != 0:
            replacements.append((matrix[i + 1][j], "up"))

    return replacements
def read_matrix_from_file(filename):
   
    # Open the file in read mode using a context manager
    with open(filename, "r") as file:
        # Read all lines from the file into a list
        lines = file.readlines()

    # Initialize an empty list to store the matrix
    matrix = []

    # Process each line in the file
    for line in lines:
        # Strip any leading/trailing whitespace and split the line by commas
        values = line.strip().split(",")

        # Convert each value to an integer
        row = [int(value) for value in values if value.strip()]

        # Append the row to the matrix
        matrix.append(row)

    return matrix

def replace_value(matrix, value, direction):
   
    # Find the position of the value in the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == value:
                # Replace the value based on the direction
                if direction == 'up':
                    if i > 0 and matrix[i-1][j] == 0:
                        matrix[i][j], matrix[i-1][j] = matrix[i-1][j], matrix[i][j]
                elif direction == 'down':
                    for row in range(i+1, len(matrix)):
                        if matrix[row][j] == 0:
                            matrix[i][j], matrix[row][j] = matrix[row][j], matrix[i][j]
                            return matrix  # Exit the function after one replacement
                elif direction == 'left':
                    if j > 0 and matrix[i][j-1] == 0:
                        matrix[i][j], matrix[i][j-1] = matrix[i][j-1], matrix[i][j]
                elif direction == 'right':
                    if j < len(matrix[0]) - 1 and matrix[i][j+1] == 0:
                        matrix[i][j], matrix[i][j+1] = matrix[i][j+1], matrix[i][j]

    return matrix




def print_matrix(matrix):
    
    # Get the number of rows and columns in the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Print the number of rows and columns
    print(f"{rows},{cols},")

    # Print each row of the matrix
    for row in matrix:
        print(",".join(map(str, row)))

def compare_states(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        state1 = [line.strip().split(',') for line in f1.readlines()]
        state2 = [line.strip().split(',') for line in f2.readlines()]

    # Check if dimensions are the same
    if len(state1) != len(state2) or len(state1[0]) != len(state2[0]):
        return False

    # Iterate over each position and compare the integers
    for i in range(len(state1)):
        for j in range(len(state1[i])):
            if state1[i][j] != state2[i][j]:
                return False
    
    return True 

def read_matrix_from_custom_format(filename):
    with open(filename, "r") as file:
        # Read all lines from the file
        lines = file.readlines()

    # Extract the dimensions from the first line
    dimensions = lines[0].strip().split(",")
    rows = int(dimensions[0])
    cols = int(dimensions[1])

    # Initialize an empty matrix
    matrix = []

    # Process the remaining lines to populate the matrix
    for line in lines[1:]:
        # Skip empty lines
        if not line.strip():
            continue
        row = []
        for value in line.strip().split(","):
            value = value.strip()
            if value:  # Check if the value is not empty
                try:
                    row.append(int(value))
                except ValueError:
                    # Handle non-integer values here
                    # For example, you can treat 'f' as a special symbol
                    row.append(value)  # Treat 'f' as a special symbol
        matrix.append(row)

    return matrix




def swap_idx(matrix, idx1, idx2):
   
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == idx1:
                matrix[i][j] = idx2
            elif matrix[i][j] == idx2:
                matrix[i][j] = idx1

def normalize_state(matrix):
    next_idx = 3
    for j in range(len(matrix)):  # Iterate over rows
        for i in range(len(matrix[0])):  # Iterate over columns
            if matrix[j][i] == next_idx:
                next_idx += 1
            elif matrix[j][i] > next_idx:
                swap_idx(matrix, next_idx, matrix[j][i])
                next_idx += 1
    return matrix


def print_matrix_in_custom_format(matrix):
   
    # Print the dimensions of the matrix
    print(f"{len(matrix[0])},{len(matrix)},")

    # Print each row of the matrix
    for row in matrix:
        print(",".join(map(str, row)))

import random

# Define the functions for generating moves, executing them, and normalizing the game state

def random_move(matrix):
  
    zero_positions = find_zeros(matrix)
    replacements = possible_replacements(matrix, zero_positions)
    return random.choice(replacements)

def execute_random_move(matrix):
   
    move = random_move(matrix)
    number, direction = move
    return replace_value(matrix, number, direction)

def random_walk(filename, N):
   
    matrix = read_matrix_from_file(filename)
    print_matrix(matrix)

    moves = 0
    while moves < N:
        print(f"\nMove {moves + 1}:")
        move = random_move(matrix)
        print(move)
        matrix = execute_random_move(matrix)
        print_matrix(matrix)
        moves += 1

        if is_puzzle_solved(filename):
            print("Goal reached!")
            break

    return matrix

def bfs(filename):
    start_time = time.time()
    explored = set() 
    frontier = []    
    moves = []     

   
    initial_state = read_matrix_from_file(filename)
    frontier.append((initial_state, [])) 

   
    while frontier:
        state, path = frontier.pop(0)  
        
   
        state_str = str(state)
        
        
        if state_str in explored:
            continue

        explored.add(state_str)  

       
        if is_puzzle_solved(filename):  
            end_time = time.time()
            solution_length = len(path)
            print("Moves:")
            for move in path:
                print(move)
            print("Final State:")
            print_matrix(state)
            print("Nodes Explored:", len(explored))
            print("Time:", "{:.2f}".format(end_time - start_time))
            print("Solution Length:", solution_length)
            return

        zero_positions = find_zeros(state)
        replacements = possible_replacements(state, zero_positions)

        
        for number, direction in replacements:
            new_state = replace_value(state, number, direction)
            frontier.append((new_state, path + [(number, direction)])) 




def dfs(filename):
    start_time = time.time()
    explored = set()  
    frontier = []     
    moves = []       
    depth_limit = 100  


    initial_state = read_matrix_from_file(filename)
    frontier.append((initial_state, [])) 

  
    while frontier:
        state, path = frontier.pop() 
        
  
        state_str = str(state)
        
        
        if state_str in explored or len(path) >= depth_limit:
            continue

        explored.add(state_str) 

 
        if is_puzzle_solved(filename):  
            end_time = time.time()
            solution_length = len(path)
            print("Moves:")
            for move in path:
                print(move)
            print("Final State:")
            print_matrix(state)
            print("Nodes Explored:", len(explored))
            print("Time:", "{:.2f}".format(end_time - start_time))
            print("Solution Length:", solution_length)
            return

        zero_positions = find_zeros(state)
        replacements = possible_replacements(state, zero_positions)

      
        for number, direction in replacements:
            new_state = replace_value(state, number, direction)
            frontier.append((new_state, path + [(number, direction)]))  


    print("No solution found within depth limit.")
    
def ids(filename):
    start_time = time.time()
    max_depth = 1  

    while True:
        explored = set() 
        frontier = []    
        moves = []      
        depth_limit = max_depth  


        initial_state = read_matrix_from_file(filename)
        frontier.append((initial_state, [])) 

        while frontier:
            state, path = frontier.pop()  

            
            state_str = str(state)

   
            if state_str in explored or len(path) >= depth_limit:
                continue

            explored.add(state_str) 


            if is_puzzle_solved(filename): 
                end_time = time.time()
                solution_length = len(path)
                print("Moves:")
                for move in path:
                    print(move)
                print("Final State:")
                print_matrix(state)
                print("Nodes Explored:", len(explored))
                print("Time:", "{:.2f}".format(end_time - start_time))
                print("Solution Length:", solution_length)
                return

            zero_positions = find_zeros(state)
            replacements = possible_replacements(state, zero_positions)

       
            for number, direction in replacements:
                new_state = replace_value(state, number, direction)
                frontier.append((new_state, path + [(number, direction)]))  

        max_depth += 1


        if max_depth > 100: 
            print("No solution found within maximum depth limit.")
            break


def heuristic(state):

    master_brick_position = None
    goal_position = None
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 1:  
                master_brick_position = (i, j)
            elif state[i][j] == 0:  
                goal_position = (i, j)

    if master_brick_position is None or goal_position is None:
        return float('inf')  
    else:
        return abs(master_brick_position[0] - goal_position[0]) + abs(master_brick_position[1] - goal_position[1])


def astar(filename):
    start_time = time.time()
    explored = set() 
    frontier = []    
    moves = []      


    initial_state = read_matrix_from_file(filename)
    initial_state_str = str(initial_state)
    initial_cost = 0
    frontier.append((initial_cost + heuristic(initial_state), initial_cost, initial_state, []))  

    while frontier:
        frontier.sort()  
        _, cost, state, path = frontier.pop(0) 
        state_str = str(state) 


        if state_str in explored:
            continue

        explored.add(state_str) 

        # Check if puzzle is solved
        if is_puzzle_solved(filename):
            end_time = time.time()
            solution_length = len(path)
            print("Moves:")
            for move in path:
                print(move)
            print("Final State:")
            print_matrix(state)
            print("Nodes Explored:", len(explored))
            print("Time:", "{:.2f}".format(end_time - start_time))
            print("Solution Length:", solution_length)
            return

        zero_positions = find_zeros(state)
        replacements = possible_replacements(state, zero_positions)


        for number, direction in replacements:
            new_state = replace_value(state, number, direction)
            new_cost = cost + 1 
            frontier.append((new_cost + heuristic(new_state), new_cost, new_state, path + [(number, direction)]))  


    print("No solution found.")


def handle_commands():
   
    args = sys.argv[1:]

    if not args:
        print("No command provided.")
        return

    command = args[0]

    if command == 'print' and len(args) == 2:
        filename = args[1]
        print_state(filename)
    
    elif command == 'done' and len(args) == 2:
        filename = args[1]
        if is_puzzle_solved(filename):
            print("True")
        else:
            print("False")

    elif command == 'availableMoves' and len(args) == 2:
        filename = args[1]
        matrix = read_matrix_from_file(filename)
        matrix = matrix[1:]
        matrix = matrix[1:]
        matrix = matrix[:-1]
        matrix = [[x for x in row if x != 1 and x != -1] for row in matrix]
        zero_positions = find_zeros(matrix)
        replacements = possible_replacements(matrix, zero_positions)
        for number, direction in replacements:
            print('('f"{number} ,{direction}"')')
     
    elif command == 'applyMove' and len(args) == 3:
        filename = args[1]
        move_str = args[2]
        move_str = move_str.strip("()").split(",")
        number = int(move_str[0])
        direction = move_str[1].strip()
        with open(filename, 'r') as file:
  
            content = file.read()
            original_state = content
        matrix = read_matrix_from_file(filename)
        matrix = matrix[1:]
        updated_matrix = replace_value(matrix, number, direction)
        print(updated_matrix)
        print_matrix(updated_matrix)

    elif command == 'compare' and len(args) == 3:
        file1 = args[1]
        file2 = args[2]
        result = compare_states(file1, file2)
        print(result)
    
    elif command == 'norm' and len(args) == 2:
        filename = args[1]
        matrix = read_matrix_from_custom_format(filename)
        normalized_matrix = normalize_state(matrix)
        print_matrix_in_custom_format(normalized_matrix)

    elif command == 'random' and len(args) == 3:
        filename = args[1]
        N = int(args[2])
        random_walk(filename, N)
    elif command == 'bfs' and len(args) == 2:
        filename = args[1]
        bfs(filename)

    elif command == 'dfs' and len(args) == 2:
        
        filename = args[1]
        dfs(filename)    

    
    elif command == 'ids' and len(args) == 2:
        
        filename = args[1]
        dfs(filename) 

    elif command == 'astar' and len(args) == 2:
        filename = args[1]
        astar(filename)           
     



if __name__ == "__main__":
    handle_commands()


