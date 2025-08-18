import PodSixNet.Channel
import PodSixNet.Server
from time import sleep
import random

class ClientChannel(PodSixNet.Channel.Channel):
		def Network(self, data):
			mServer.makeMove(data)

class StrategioServer(PodSixNet.Server.Server):
 
		channelClass = ClientChannel
		
		def __init__(self, *args, **kwargs):
			PodSixNet.Server.Server.__init__(self, *args, **kwargs)
			self.games = []
			self.queue = None
			self.currentIndex = 0

		def makeMove(self, data):
			player = data["player"]
			gameid = data["gameid"]
			print data
			print "player :" + str(player) + " moved"
			self.games[gameid].perform_attack(data['attack_pos'], data['target_pos'])
			self.games[gameid].player1.Send({"action" : "updateboard", "board" : self.games[gameid].board})
			self.games[gameid].player2.Send({"action" : "updateboard", "board" : self.games[gameid].board})

		def Connected(self, channel, addr):
			if(self.queue == None):
				print "waiting game index :" + str(self.currentIndex)
				channel.gameid = self.currentIndex
				self.queue = Game(self.currentIndex)
				self.queue.player1 = channel
				self.currentIndex +=1
			else :
				print "starting game index :" + str(self.currentIndex)
				channel.gameid = self.currentIndex
				self.queue.player2 = channel
				self.queue.initBoard()
				self.queue.player1.Send({"action": "start","player":1, "gameid": self.queue.gameid, "board" : self.queue.board})
				self.queue.player2.Send({"action": "start","player":2, "gameid": self.queue.gameid, "board" : self.queue.board})
				self.games.append(self.queue)
				self.queue = None
			print 'new connection:', channel

class Game :
	def __init__(self, game_index):
		self.player1 = None
		self.player2 = None
		self.gameid  = game_index

	def perform_attack(self,attack_pos, target_pos):
			
			if (self.is_valid_move(attack_pos, target_pos)):
				# Check who will win the attack
				if(self.board[attack_pos[0]][attack_pos[1]][1] == self.board[target_pos[0]][target_pos[1]][1]):
					self.board[target_pos[0]][target_pos[1]] = (-1,-1)			
				elif(self.board[attack_pos[0]][attack_pos[1]][1] > self.board[target_pos[0]][target_pos[1]][1]):
					self.board[target_pos[0]][target_pos[1]] = self.board[attack_pos[0]][attack_pos[1]]
				# Leave attack position
				self.board[attack_pos[0]][attack_pos[1]] = (-1,-1)

	def is_valid_move(self, attack_pos, target_pos) :
		attack_piece = self.board[attack_pos[0]][attack_pos[1]]
		target_piece = self.board[target_pos[0]][target_pos[1]]

		# Player can just move one move each time
		if(abs(attack_pos[0] - target_pos[0]) > 1 or abs(attack_pos[1] - target_pos[1]) > 1) :
			return False
		elif(attack_piece[0] == target_piece[0]) :# Player can't attack its place
			return False
		elif(target_piece[0] == 0) : # Player can't attack a lake place
			return False
		else :
			return True


	def initBoard(self):
		board = [[0 for x in range(10)] for y in range(10)]
		
		# Randomly initialize first user pieces
		for i in range(0,3):
			for j in range(10):
				board[i][j] = (1,random.choice([1,2,3,4,5,6]))
		
		# Set the middle pieces to empty
		for i in range(3,7):
			for j in range(10):
				board[i][j] = (-1,-1)	
		
		# Set two middle lakes
		for i in range(4,6) :
			for j in range(2,4):
				board[i][j] = (0,-1)	
			for k in range(6,8) :
				board[i][k] = (0,-1)	

		# Randomly initialize second user pieces
		for i in range(7,10):
			for j in range(10):
				board[i][j] = (2,random.choice([1,2,3,4,5,6]))
		self.board = board


mServer = StrategioServer()
while True:
    mServer.Pump()
    sleep(0.01)