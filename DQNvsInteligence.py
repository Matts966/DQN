import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

#ゲームボード
class Board():
    def reset(self):
        self.board = np.array([0] * 9, dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False

    def move(self, act, turn):
        if self.board[act] == 0:
            self.board[act] = turn
            self.check_winner()
        else:
            self.winner = turn*-1
            self.missed = True
            self.done = True

    def check_winner(self):
        win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for cond in win_conditions:
            if self.board[cond[0]] == self.board[cond[1]] == self.board[cond[2]]:
                if self.board[cond[0]] != 0:
                    self.winner = self.board[cond[0]]
                    self.done = True
                    return
        if np.count_nonzero(self.board) == 9:
            self.winner = 0
            self.done = True

    def get_empty_pos(self):
        empties = np.where(self.board==0)[0]
        if len(empties) > 0:
            return np.random.choice(empties)
        else:
            return 0

    def show(self):
        row = " {} | {} | {} "
        hr = "\n-----------\n"
        tempboard = []
        count = 0
        for i in self.board:
            count += 1
            if i == 1:
                tempboard.append("○")
            elif i == -1:
                tempboard.append("×")
            else:
                tempboard.append(str(count))
        print((row + hr + row + hr + row).format(*tempboard))
        print("\n")


# ボードの準備
b = Board()

# MiniMax法
class MiniMax:
    # def minimax21(self, board, level, alpha, beta):
    #     winner = board.check_winner()   
	# 	if winner:
    #         return 

	# 	if level == 0:
	# 		# TODO: normalize this between -1 and 1. Also come up with a better way to identify player/computer symbols
	# 		return -1 * board.evaluate(self.player_symbol), None

	# 	states_and_moves = board.next_states_and_moves()
	# 	best_move = None
	# 	if comp_turn:
	# 		for state, move in states_and_moves:
	# 			score = self.minimax(state, level - 1, False, alpha, beta)[0]
	# 			if score > alpha:
	# 				alpha = score
	# 				best_move = move
	# 			if alpha >= beta:
	# 				break
	# 		return alpha, best_move
	# 	else:
	# 		for state, move in states_and_moves:
	# 			score = self.minimax(state, level - 1, True, alpha, beta)[0]
	# 			if score < beta:
	# 				beta = score
	# 				best_move = move
	# 			if alpha >= beta:
	# 				break
	# 		return beta, best_move

    
    def minimax(self, copiedBoard, mini_first):
        win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for cond in win_conditions:
            if copiedBoard[cond[0]] == copiedBoard[cond[1]] == copiedBoard[cond[2]]:
                if copiedBoard[cond[0]] == 1:
                    return -10
                elif copiedBoard[cond[0]] == -1:
                    return 10

        empties = np.where(copiedBoard == 0)[0]

        if len(empties) == 0:
            return 0

        children = []
        minimaxedChildren = []

        # 相手の番
        if np.sum(copiedBoard) < 0 or (not mini_first and np.sum(copiedBoard) == 0):
            for e in empties:
                cb = copiedBoard.copy()
                cb[e] = 1
                children.append(cb)
            for child in children:
                minimaxedChildren.append(self.minimax(child, mini_first))
            return min(minimaxedChildren)

        # Minimaxの番
        else:
            for e in empties:
                cb = copiedBoard.copy()
                cb[e] = -1
                children.append(cb)
            for child in children:
                minimaxedChildren.append(self.minimax(child, mini_first))
            return max(minimaxedChildren)
                

    def act(self, board, mini_first):
        empties = np.where(board==0)[0]
        if len(empties) == 9:
            return np.random.choice(empties)
        if len(empties) == 0:
            print(board)
            print('Bug')
            return 0
        if len(empties) == 1:
            return empties[0]
        children = []
        for e in empties:
            b = board.copy()
            b[e] = -1
            m = self.minimax(b, mini_first)
            children.append(m)
        return empties[children.index(max(children))]      

class intelligentMethod:
    # 今の状態から次にどこに置くかを決定する
    def act(self, board):
        
        empties = np.where(board==0)[0]
        win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

        # もし勝てるのであれば勝つ
        for e in empties:
            b = board.copy()
            b[e] = -1
            for cond in win_conditions:
                if b[cond[0]] == b[cond[1]] == b[cond[2]]:
                    if b[cond[0]] == -1:
                        return e

        # もし負けるのであれば妨害を入れる
        for e in empties:
            b = board.copy()
            b[e] = 1
            for cond in win_conditions:
                if b[cond[0]] == b[cond[1]] == b[cond[2]]:
                    if b[cond[0]] == 1:
                        return e

        # もし中央が空いていればそこに置く.
        if board[4] == 0:
            return 4

        # もし隅が空いていれば隅の適当な場所に置く.
        corner = (0, 2, 6, 8)
        for c in corner:
            if board[c] == 0:
                return c

        # それ以外の場合は適当に置く.
        for c in range(9):
            if board[c] == 0:
                return c


# MiniMaxとIntelの勝負で検証
IntM = intelligentMethod()
mini_max = MiniMax()
mini = 0
Int = 0
draw = 0
for i in range(10):
    b.reset()
    Int_first = np.random.choice([True, False])
    while not b.done:
        #Intel
        if Int_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = IntM.act(b.board.copy())
            b.move(action, 1)
            if b.done == True:
                b.show()
                if b.winner == 1:
                    print("IntM Win")
                    Int += 1
                elif b.winner == 0:
                    print("Draw")
                    draw += 1
                else:
                    print("DQN Missed??????????BUG!!!!!!!!!!")
                    print('action : ' + str(action))
                    mini += 1 
                continue
        #Minimax
        b.show()
        action = mini_max.act(b.board.copy(), not Int_first)
        b.move(action, -1)
        if b.done == True:
            b.show()
            if b.winner == -1:
                print("MiniMax Win")
                mini += 1
            elif b.winner == 0:
                print("Draw")
                draw += 1
print ('Int WIN : ' + str(Int) + ' Mini WIN : ' + str(mini) + ' Draw : ' + str(draw))

