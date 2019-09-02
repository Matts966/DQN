import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

# 今回作成したプログラムは、三目並べを解くことを目的として、DoubleDQN（Deep Q Network）と
# 呼ばれる学習手法を、ChainerRLというライブラリを用いて実装し、Minimax法と簡単なコツを利用
# したアルゴリズムと戦わせる、というものである。
# 30万回強の学習回数でも、学習は収束せず、簡単なアルゴリズムにもまけることがある、ということがわかった。
# ただ、そもそも置けない場所を選択肢に含め、それを選んだ場合にMissとして処理するという計測方法
# は不適切だったかもしれない。
# 機械学習も、目的や細かい実装によっては、思い通りの結果を出すことができない、ということの例には
# なるかもしれない。
# 最後には、実際に自分の手で遊んで、学習の度合を確かめられるようになっている。
# 以下はそのプログラムである。

#三目並べ用のボードをクラスとして実装
class Board():
    def reset(self):
        self.board = np.array([0] * 9, dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False

    def move(self, action, turn):
        if self.board[action] == 0:
            self.board[action] = turn
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

#explorer用のランダム関数オブジェクト
class RandomActor:
    def __init__(self, board):
        self.board = board
        self.random_count = 0
    def random_action_func(self):
        self.random_count += 1
        return self.board.get_empty_pos()

#Q関数
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels = 81):
        super().__init__(
            l0 = L.Linear(obs_size, n_hidden_channels),
            l1 = L.Linear(n_hidden_channels, n_hidden_channels),
            l2 = L.Linear(n_hidden_channels, n_hidden_channels),
            l3 = L.Linear(n_hidden_channels, n_actions))
    def __call__(self, x, test=False):
        #-1を扱うのでleaky_reluとした
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

# ボードの準備
b = Board()

# explorer用のランダム関数オブジェクトの準備
ra = RandomActor(b)

# 環境と行動の次元数
obs_size = 9
n_actions = 9

# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# 報酬の割引率
gamma = 0.95

# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon = 1.0, end_epsilon = 0.3, decay_steps = 50000
    , random_action_func = ra.random_action_func)

# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10 ** 6)

# Agentの生成（replay_buffer等を共有する2つ）
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size = 500, update_interval = 1,
    target_update_interval = 100)

agent_p2 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size = 500, update_interval = 1,
    target_update_interval = 100)

# Modelのロード。最初から学習させたければコメントアウト、保存されたモデルを再利用してスタートしたい場合はアンコメントアウト。
# agent_p1.load('dqn_result')
# agent_p2.load('dqn_result')

#学習ゲーム回数
#ほんとはたくさん稼ぎたいが、実際のところメモリと所要時間的に辛い
print('学習回数を入力 > ', end='')
n_episodes = int(input())

#カウンタの宣言
miss = 0
win = 0
draw = 0
#エピソードの繰り返し実行
for i in range(1, n_episodes + 1):
    b.reset()
    reward = 0
    agents = [agent_p1, agent_p2]
    turn = np.random.choice([0, 1])
    last_state = None
    while not b.done:
        #配置マス取得
        action = agents[turn].act_and_train(b.board.copy(), reward)
        #配置を実行
        b.move(action, 1)
        #配置の結果、終了時には報酬とカウンタに値をセットして学習
        if b.done == True:
            if b.winner == 1:
                reward = 1
                win += 1
            elif b.winner == 0:
                draw += 1
            else:
                reward = -1
            if b.missed is True:
                miss += 1
            #エピソードを終了して学習
            agents[turn].stop_episode_and_train(b.board.copy(), reward, True)
            #相手もエピソードを終了して学習。相手のミスは勝利として学習しないように
            if agents[1 if turn == 0 else 0].last_state is not None and b.missed is False:
                #前のターンでとっておいたlast_stateをaction実行後の状態として渡す
                agents[1 if turn == 0 else 0].stop_episode_and_train(last_state, reward * -1, True)
        else:
            #学習用にターン最後の状態を退避
            last_state = b.board.copy()
            #継続のときは盤面の値を反転
            b.board = b.board * -1
            #ターンを切り替え
            turn = 1 if turn == 0 else 0

    #コンソールに進捗表示
    if i % 100 == 0:
        print("episode:", i, " / rnd:", ra.random_count, " / miss:", miss
        , " / win:", win, " / draw:", draw, " / statistics:", agent_p1.get_statistics()
        , " / epsilon:", agent_p1.explorer.epsilon)
        #カウンタの初期化
        miss = 0
        win = 0
        draw = 0
        ra.random_count = 0
    if i % 10000 == 0:
        # 10000エピソードごとにモデルを保存
        agent_p1.save("dqn_result")

print("Training finished.")


#人間のプレーヤー
class HumanPlayer:
    def act(self, board):
        valid = False
        while not valid:
            try:
                act = input("Please enter 1-9: ")
                act = int(act)
                if act >= 1 and act <= 9 and board[act-1] == 0:
                    valid = True
                    return act-1
                else:
                    print("Invalid move")
            except Exception as e:
                print(act +  " is invalid")



# MiniMax法
class MiniMax:
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

# 負けなし戦略ではないがそこそこ強い戦略
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

        # もし隅が空いていればランダムに隅の場所に置く.
        corner = (0, 2, 6, 8)
        for c in corner:
            if board[c] == 0:
                return c

        # それ以外の場合は適当に置く.
        for c in range(9):
            if board[c] == 0:
                return c


# MiniMaxとDQNの勝負で検証
mini_max = MiniMax()
mini = 0
dqn = 0
draw = 0
for i in range(10):
    b.reset()
    dqn_first = np.random.choice([True, False])
    while not b.done:
        #DQN
        if dqn_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = agent_p1.act(b.board.copy())
            b.move(action, 1)
            if b.done == True:
                b.show()
                if b.winner == 1:
                    print("DQN Win")
                    dqn += 1
                elif b.winner == 0:
                    print("Draw")
                    draw += 1
                else:
                    print("DQN Missed")
                    print('action : ' + str(action))
                    mini += 1 
                agent_p1.stop_episode()
                continue
        #Minimax
        b.show()
        action = mini_max.act(b.board.copy(), not dqn_first)
        b.move(action, -1)
        if b.done == True:
            b.show()
            if b.winner == -1:
                print("MiniMax Win")
                mini += 1
            elif b.winner == 0:
                print("Draw")
                draw += 1
            agent_p1.stop_episode()
print ('DQN WIN : ' + str(dqn) + ' MiniMax WIN : ' + str(mini) + ' Draw : ' + str(draw))

# ちょっとした原則に基づいたプレイと比較
IntM = intelligentMethod()
Int = 0
dqn = 0
draw = 0
for i in range(10):
    b.reset()
    dqn_first = np.random.choice([True, False])
    while not b.done:
        #DQN
        if dqn_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = agent_p1.act(b.board.copy())
            b.move(action, 1)
            if b.done == True:
                b.show()
                if b.winner == 1:
                    print("DQN Win")
                    dqn += 1
                elif b.winner == 0:
                    print("Draw")
                    draw += 1
                else:
                    print("DQN Missed")
                    print('action : ' + str(action))
                    Int += 1 
                agent_p1.stop_episode()
                continue
        #Minimax
        b.show()
        action = IntM.act(b.board.copy())
        b.move(action, -1)
        if b.done == True:
            b.show()
            if b.winner == -1:
                print("Int Win")
                Int += 1
            elif b.winner == 0:
                print("Draw")
                draw += 1
            agent_p1.stop_episode()
print ('DQN WIN : ' + str(dqn) + ' Int WIN : ' + str(Int) + ' Draw : ' + str(draw))

#人間との勝負で検証
human_player = HumanPlayer()
for i in range(10):
    b.reset()
    dqn_first = np.random.choice([True, False])
    while not b.done:
        #DQN
        if dqn_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = agent_p1.act(b.board.copy())
            b.move(action, 1)
            if b.done == True:
                b.show()
                if b.winner == 1:
                    print("DQN Win")
                elif b.winner == 0:
                    print("Draw")
                else:
                    print("DQN Missed")
                    print('action:' + str(action))
                agent_p1.stop_episode()
                continue
        #人間
        b.show()
        action = human_player.act(b.board.copy())
        b.move(action, -1)
        if b.done == True:
            b.show()
            if b.winner == -1:
                print("HUMAN Win")
            elif b.winner == 0:
                print("Draw")
            agent_p1.stop_episode()

#三十五万回程度では収束せず。。
print("Test finished.")
