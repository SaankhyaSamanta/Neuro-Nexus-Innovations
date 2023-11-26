import pygame
import random

# Initialize Pygame
pygame.init()

# Set the dimensions of the game window
Width, Height = 600, 600
WINDOW_SIZE = (Width, Height)

# Set colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set the font for displaying text
FONT = pygame.font.Font(None, 200)

# Create the game window
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Tic Tac Toe")

# Set the size of each square on the game board
BoardSize = 3  # Size of the board (BoardSize x BoardSize)
SquareSize = Width // BoardSize

# Create the game board
board = [[" " for _ in range(BoardSize)] for _ in range(BoardSize)]
players = ["X", "O"]
Player = random.choice(players)
GameOver = False

# Create a flag to control the game loop
Play = True

def DrawBoard():
    # Clear the window
    window.fill(WHITE)

    # Draw the vertical lines
    for i in range(1, BoardSize):
        pygame.draw.line(window, BLACK, (i * SquareSize, 0), (i * SquareSize, Height), 3)

    # Draw the horizontal lines
    for i in range(1, BoardSize):
        pygame.draw.line(window, BLACK, (0, i * SquareSize), (Width, i * SquareSize), 3)

    # Draw the X and O markers on the board
    for row in range(BoardSize):
        for col in range(BoardSize):
            if board[row][col] == "X":
                text = FONT.render("X", True, BLACK)
                window.blit(text, (col * SquareSize + 60, row * SquareSize + 60))
            elif board[row][col] == "O":
                text = FONT.render("O", True, BLACK)
                window.blit(text, (col * SquareSize + 60, row * SquareSize + 60))

    # Update the display
    pygame.display.update()

#Check if draw
def BoardFull():
    for i in range(BoardSize):
        for j in range(BoardSize):
            if board[i][j]==" ":
                return False
    return True

# Check if a player has won
def Winner():
    # Check rows
    for row in range(BoardSize):
        if all(board[row][j]=="X" for j in range(BoardSize)):
            return "X"
        elif all(board[row][j]=="O" for j in range(BoardSize)):
            return "O"
    # Check columns
    for col in range(BoardSize):
        if all(board[i][col]=="X" for i in range(BoardSize)):
            return "X"
        elif all(board[i][col]=="O" for i in range(BoardSize)):
            return "O"

    # Check diagonals
    if all(board[i][i] == "X" for i in range(BoardSize)):
        return "X"
    elif all(board[i][i] == "O" for i in range(BoardSize)):
        return "O"
    elif all(board[i][BoardSize - 1 - i] == "X" for i in range(BoardSize)):
        return "X"
    elif all(board[i][BoardSize - 1 - i] == "O" for i in range(BoardSize)):
        return "O"

    return None

# The Minimax algorithm for determining the best move
def Minimax(board, depth, ai_player):
    winner = Winner()
    if winner=="X":
        return -1
    elif winner=="O":
        return 1
    elif BoardFull():
        return 0

    if ai_player:
        maxscore = float("-inf")
        for row in range(BoardSize):
            for col in range(BoardSize):
                if board[row][col] == " ":
                    board[row][col] = "O"
                    score = Minimax(board, depth + 1, False)
                    board[row][col] = " "
                    maxscore = max(score, maxscore)
        return maxscore
    else:
        minscore = float("inf")
        for row in range(BoardSize):
            for col in range(BoardSize):
                if board[row][col] == " ":
                    board[row][col] = "X"
                    score = Minimax(board, depth + 1, True)
                    board[row][col] = " "
                    minscore = min(score, minscore)
        return minscore

# Get the best move for the AI player using the Minimax algorithm
def AIMove(board):
    bestscore = float("-inf")
    best_move = None
    for row in range(BoardSize):
        for col in range(BoardSize):
            if board[row][col] == " ":
                board[row][col] = "O"
                score = Minimax(board, 0, False)
                board[row][col] = " "
                if score > bestscore:
                    bestscore = score
                    best_move = (row, col)
    return best_move

# Game loop
while Play:
    # Reset the game variables
    board = [[" " for _ in range(BoardSize)] for _ in range(BoardSize)]
    Player = random.choice(players)
    GameOver = False
    DrawBoard()
    # Inner game loop
    while not GameOver:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                GameOver = True

            if event.type == pygame.MOUSEBUTTONDOWN and not GameOver and Player == "X":
                x, y = pygame.mouse.get_pos()
                row = y // SquareSize
                col = x // SquareSize
                if board[row][col] == " ":
                    board[row][col] = "X"
                    DrawBoard()
                    winner = Winner()
                    if winner == "X":
                        print("You won!")
                        GameOver = True
                    elif BoardFull():
                        print("It's a draw!")
                        GameOver = True
                    Player = "O"

            if not GameOver and Player == "O":
                print("AI's Turn")
                best_move = AIMove(board)
                if best_move:
                    row, col = best_move
                    board[row][col] = "O"
                    DrawBoard()
                    winner = Winner()
                    if winner == "O":
                        print("AI won!")
                        GameOver = True
                    elif BoardFull():
                        print("It's a draw!")
                        GameOver = True
                Player = "X"

        

    # Ask the player if they want to play again
    Play = int(input("Enter 1 to play again, 0 to exit: "))
    if Play==False:
        pygame.quit()

# Quit Pygame
pygame.quit()

