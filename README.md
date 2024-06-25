# matrix-based-puzzles

This Python script offers a collection of functions designed to manipulate and solve puzzles represented as matrices in text files. The script can perform a variety of operations, including reading and modifying the puzzle state, checking if the puzzle is solved, and applying different search algorithms to find solutions.

## Features

- **State Display**: Display the current state of the puzzle.
- **Solve Check**: Check if the puzzle is solved.
- **Move Simulation**: Simulate possible moves from the current puzzle state.
- **Move Application**: Apply a specific move to the puzzle.
- **State Comparison**: Compare two puzzle states to check if they are identical.
- **Matrix Normalization**: Normalize the puzzle matrix to a standard format.
- **Random Walk**: Perform a random walk through the puzzle state.
- **Search Algorithms**: Includes Breadth-First Search (BFS), Depth-First Search (DFS), Iterative Deepening Search (IDS), and A* search to solve the puzzle.

## Functions

- `print_state(filename)`: Prints the current state of the puzzle from a file.
- `is_puzzle_solved(filename)`: Returns True if the puzzle is solved.
- `find_zeros(board)`: Finds positions of zeros in the matrix.
- `possible_replacements(matrix, zero_positions)`: Lists possible replacements based on zero positions.
- `read_matrix_from_file(filename)`: Reads a matrix from a given file.
- `replace_value(matrix, value, direction)`: Replaces a matrix value based on a specified direction.
- `print_matrix(matrix)`: Prints the matrix.
- `compare_states(file1, file2)`: Compares two matrix states.
- `normalize_state(matrix)`: Normalizes the matrix to a standard format.
- `random_walk(filename, N)`: Executes a random walk on the matrix.
- `bfs(filename)`: Solves the puzzle using Breadth-First Search.
- `dfs(filename)`: Solves the puzzle using Depth-First Search.
- `ids(filename)`: Solves the puzzle using Iterative Deepening Search.
- `astar(filename)`: Solves the puzzle using A* search based on a heuristic.

## Usage

To use the script, run it with the desired command and arguments from the command line:

```bash
python puzzle_solver.py [command] [arguments]
