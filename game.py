#Sets up the environment for the Snake game, but without the human controls and modded to return a reward instead of ending.

import pygame
import random
import enum
import numpy as np
from collections import namedtuple

pygame.init()
#setup font for score display (use system font for portability).
font = pygame.font.SysFont('arial', 25)

#Enum makes reading code easier (Right=1, Left=2, etc.
class Direction(enum.Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#A simple wrapper for coordinates so we can say point.x and point.y instead of point[0]
Point = namedtuple('Point', 'x, y')

#Constants
BLOCK_SIZE = 20
SPEED = 100000  #speed of the game (Higher = faster training)

class SnakeGameAI:
    def __init__(self, w=640, h=480, visual=True):
        self.w = w
        self.h = h
        self.visual = visual
        #init Pygame window
        if self.visual:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None
        self.reset()

    def reset(self):
        #Reset the game state for a new training episode
        self.direction = Direction.RIGHT

        #start in the middle of the screen
        self.head = Point(self.w/2, self.h/2)

        #Create Initial body
        self.snake = [self.head,
                        Point(self.head.x - BLOCK_SIZE, self.head.y),
                        Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # keeps track of how long the game has lasted 

    def _place_food(self):
        # pick a random spot on the grid
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        # Recursive check: If food spawns on the snake, put  it somewhere else
        if self.food in self.snake:
            self._place_food()

    # Most important function
    #The aent calls this function to amke a move
    #it sends an 'action' (ex: [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn)
    #the game returns: Reward (Math), Game over (bool), Score(Display)
    def play_step(self, action):
        self.frame_iteration += 1
        #1. Handle Window Events(ALlows you to click X to close)
        if self.visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        # Distance to food before move (Manhattan)
        dist_old = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        #2. Move the snake
        self._move(action) #updates the head pos based on direction
        self.snake.insert(0, self.head) #add new head to the snake body

        #3. Check if game over
        reward = 0
        game_over = False

        #Collision Check:
        #if hits wall, hits self, or takes too long (starvation protection)
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -50 #Big penalty for dying
            return reward, game_over, self.score
        
        #4. Check food
        if self.head == self.food:
            self.score += 1
            reward = 50 # big reward for eating food
            self._place_food()
        else:
            #If snake didn't eat food, remove the tail (snake moves forward)
            self.snake.pop() 
            # Reward shaping: small reward for moving closer to food
            dist_new = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if dist_new < dist_old:
                reward += 0.1
            elif dist_new > dist_old:
                reward -= 0.1
        
        #5. Update UI and clock
        if self.visual:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        #Check boundry collision (hit the walls)
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        #Check body collision (hits self)
        if pt in self.snake[1:]:
            return True
        return False
    
    def _update_ui(self):
        if not self.visual:
            return
        self.display.fill((0,0,0)) #fill screen with black

        # Draw Snake (Green)
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Food (Red)
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Score text
        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip() # Update full display

    def _move(self, action):
        # Action comes in as a vector:
        # [1, 0, 0] -> Straight
        # [0, 1, 0] -> Right Turn
        # [0, 0, 1] -> Left Turn
        
        # Order of directions (Clockwise)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4 # Move to next index (Right turn)
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4 # Move to previous index (Left turn)
            new_dir = clock_wise[next_idx]

        self.direction = new_dir
        
        # Update coordinates based on direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        
        self.head = Point(x, y)