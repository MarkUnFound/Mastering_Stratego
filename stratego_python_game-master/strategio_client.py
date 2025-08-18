import pygame
import random
import math
from PodSixNet.Connection import ConnectionListener, connection
from time import sleep

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0,0,255)
 
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 40
HEIGHT = 40
 
# This sets the margin between each cell
MARGIN = 5

class MyStratego(ConnectionListener):
	"""docstring for ClassName"""
	def __init__(self):
		pygame.init()
		self.gameid = None
		self.num 		= None
		self.screen = pygame.display.set_mode((455,500))
		self.clock  = pygame.time.Clock()
		self.font   = pygame.font.SysFont('Aryal', 15)
		self.Connect()
		self.holdOn()

	def update(self):
		connection.Pump()
		self.Pump()
		self.clock.tick(60)
		self.screen.fill(0)
		#self.screen.blit(BackGround.image, BackGround.rect)
		self.drawGrid()

		mouse = pygame.mouse.get_pos()
		xpos = int(math.ceil((mouse[0]-25)/50.0))
		ypos = int(math.ceil((mouse[1]-25)/50.0))

		for event in pygame.event.get() :
			if event.type == pygame.QUIT:
				exit()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				self.attack_pos = (ypos, xpos)
			elif event.type == pygame.MOUSEBUTTONUP:
				self.target_pos = (ypos, xpos)
				self.Send({"action" : "move", "attack_pos" : self.attack_pos, "target_pos" : self.target_pos, "player" : self.player, "gameid" : self.gameid})
				# self.perform_attack()

		pygame.display.flip()

	def drawGrid(self) :
		for row in range(10):
			for column in range(10):
				if(self.board[row][column][0] == self.player): # Drawing the alies pieces
					pygame.draw.rect(self.screen,GREEN,[(MARGIN + WIDTH) * column + MARGIN,(MARGIN + HEIGHT) * row + MARGIN,WIDTH,HEIGHT])
					self.screen.blit(self.font.render(str(self.board[row][column][1]), True, BLACK),((MARGIN + WIDTH) * column + MARGIN + (MARGIN),(MARGIN + HEIGHT) * row + MARGIN + (MARGIN)))
				elif(self.board[row][column][0] == 0) : # Drawing the lake
					pygame.draw.rect(self.screen,BLUE,[(MARGIN + WIDTH) * column + MARGIN,(MARGIN + HEIGHT) * row + MARGIN,WIDTH,HEIGHT])
				elif(self.board[row][column][0] == -1):
					pass
				else : # Drawing enemies pieces
					pygame.draw.rect(self.screen,RED,[(MARGIN + WIDTH) * column + MARGIN,(MARGIN + HEIGHT) * row + MARGIN,WIDTH,HEIGHT])

	def holdOn(self) :
		self.go = False
		while not self.go :
			connection.Pump()
			self.Pump()
			sleep(0.01)

	def Network_start(self, data):
		self.go = True
		self.player = data["player"]
		self.gameid = data["gameid"]
		self.board  = data["board"]

	def Network_updateboard(self, data) :
		self.board = data['board']

ms = MyStratego()
while 1:
	ms.update()