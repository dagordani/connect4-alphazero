import numpy as np
import torch
import torch.nn.functional as F

FPU = 0
ROWS, COLS = 6, 7
NUM_NODES = 343
ALPHA = 0.3
EPS = 0.25

class State:
    def __init__(self, board, last_move_coordinates=None, game_ended=False, reward=None):
        self.board = board
        self.rows = board.shape[0]
        self.cols = board.shape[1]
        self.allowed_moves = np.where(board[0] == 0)[0]
        self.skyline = (board != 0.0).sum(axis=0)
        self.last_move_coordinates = last_move_coordinates
        self.game_ended = game_ended
        self.reward = reward

    @classmethod
    def from_empty_board(cls, rows=ROWS, cols=COLS):
        return cls(np.zeros((rows, cols)))

    @classmethod
    def from_state(cls, state):
        """Creates a fresh instance directly from another State instance."""
        return cls(
            board=state.board.copy(),
            last_move_coordinates=state.last_move_coordinates,
            game_ended=state.game_ended,
            reward=state.reward,
        )

    def copy_state(self):
        return State.from_state(self)
    
    def _update_board(self, action: int) -> None:
        x = int(self.rows - self.skyline[action] - 1)
        y = action
        self.last_move_coordinates = (x, y)
        self.board[x, y] = 1  # update board
        self.skyline[y] += 1  # update skyline
        self.allowed_moves = np.where(self.board[0] == 0)[0]   # update allowed moves
    
    def transition_state_in_place(self, action):
        """used once in expand(), use many times in simulate()"""
        self._update_board(action)  # make move
        self._check_winning_condition()  # update .game_ended and .reward
        self.board *= -1  # flip board to pass to other player. Call after ._check_winning_condition since that checks for +1 pieces
    
    def transition_state(self, action):
        """for expand()"""
        state = self.copy_state()  # creates state: last_move_coordinates=None and game_ended=False
        state.transition_state_in_place(action)  # updates last_move_coordinates via _update_board() and game_ended via check_winning_condition()
        return state

    def _check_winning_condition(self) -> None:
        """check if latest move ended the game through win (return 1) or draw (return 0) or None for neither"""

        # iterate from -3 to +3 position from index of latest move in all 4 directions (horizontal, vertical, both diagonals)
        # we do sliding window of 4 along 7 positions: start from 3,2,1,0 positions to left, and check the 4 right positions
        CONN = 4
        DIRS = [(0, 1), (-1, 0), (-1, 1), (1, 1)]  # right, up, up-right, down-right
        x, y = self.last_move_coordinates
        x_bound, y_bound = self.rows, self.cols
        for dx, dy in DIRS:
            for d in range(-CONN + 1, 1):
                # move in negative direction to set starting point of window
                start_x, start_y = x + d*dx, y + d*dy
                
                # skip out of bounds
                if start_x < 0 or start_x >= x_bound or start_y < 0:                    
                    continue
                    
                # move in positive direction to traverse window
                count = 0
                for i in range(CONN):
                    xx, yy = start_x + i*dx, start_y + i*dy
                    
                    # move within bounds
                    if xx >= 0 and xx < x_bound and yy >= 0 and yy < y_bound:
                        count += self.board[xx, yy] == 1
                        
                    if count == CONN:
                        # player that made last move won, but since the resulting state is of the opponent we set reward to -1
                        self.game_ended = True
                        self.reward = -1
                        return

        # board is full, and win wasn't found, therefore it's a draw
        if self.allowed_moves.size == 0:
            self.game_ended = True
            self.reward = 0
        return

    def __repr__(self):
        return f"Game_Ended? {self.game_ended}\n  board:\n{self.board}\n  skyline: {self.skyline}\n  allowed moves: {self.allowed_moves}"
        

class Game:
    def __init__(self, state: State):
        self.state = state

    def move(self, action):
        self.state.transition_state_in_place(action)


class Node:
    def __init__(self, node_id: int, state: State):
        self.id = node_id  # order of nodes created
        self.state = state
        self.n = 0  # visit count (int)
        self.w = 0.0  # cumulative reward (float)
        self.parent: Node = None
        self.children: list[tuple[int, Node]] = []
        self.prior = None  # action probs over its children assigned by policy head
        self.evaluated = False  # whether node has been through model's fwd pass

    def add_child(self, action, child):
        assert isinstance(child, Node)
        self.children.append((action, child))
        child.parent = self
        child.children = []
    
    def __repr__(self):
        return f"id: {self.id}, visit count: {self.n}, value: {self.w}, num_children: {len(self.children)}, prior {self.prior}, evaluated? {self.evaluated}\n{self.state}"


class MCTS:
    def __init__(self, root: Node, model, num_nodes=NUM_NODES, add_noise=False):
        assert root.id == 0, "root node of MCTS must have id = 0"
        self.root = root
        self.model = model
        self.num_nodes = num_nodes
        self.add_noise = add_noise
        self.size = 1

    def _select(self):
        """
        use policy to choose among all legal moves to decide which child to add
        returns first unevaluated or terminal node
        PUCT: Q(a) + C * P(a) * sqrt(N) / (1 + n(a))
        exploitation: Q = w(a)/n(a). Negative since parent wants weakest state of child since its opponent
        exploration: 
        """
        C = 1.5 # conventional value of 2 * np.sqrt(2) weights too much exploration
        node = self.root

        # go thorugh evaluated nodes (have stored prior for children) that are non-terminal (haven't ended game)
        while node.evaluated and not node.state.game_ended:            

            # calculate PUCT (Polynomial Upper Confidence bounds applied to Trees) criteria
            PUCT_criteria = np.full(COLS, -np.inf)
            for a in node.state.allowed_moves:

                # if child is in graph, use its stats, else use prior
                p = node.prior[0,a].item()
                if a in [a for (a, _) in node.children]:
                    child = [c for (i, c) in node.children if a == i][0]
                    exploitation = -1 * (child.w / child.n)
                    exploration = p * np.sqrt(node.n) / (1 + child.n)
                    criteria = exploitation + C * exploration
                else:
                    # Note: all untried moves only differ in the prior but need to include C and sqrt(N) since criteria score is compared with the tried moves
                    # Because we don't backprop on root, the first time we enter while loop node.n=root.n=0 so we do max() to just use FPU+C*p as criteria
                    criteria = FPU + C * p * np.sqrt(max(node.n, 1))

                PUCT_criteria[a] = criteria

            # choose best action
            action = PUCT_criteria.argmax()

            # add child to graph if not present, and point node variable to child
            for (a, child) in node.children:
                if action == a:
                    break
            else:
                # didn't break so action not present in graph, add it
                child = Node(self.size, node.state.transition_state(action))
                node.add_child(action, child)
                self.size += 1
            node = child

        return node
        
    def _backpropagate(self, node, val):
        """
        increment value and visit_count of all parent nodes starting from 'node' to the root
        'val' correspond to 'node' so start with positive reward alternate signs for each level/player up
        """
        cur = node
        
        # exits before root values change (root is the only node with parent = None)
        i = 0
        while cur.parent is not None:
            cur.n += 1
            cur.w += val if (i % 2 == 0) else -val
            cur = cur.parent
            i += 1
        self.root.n += 1
        self.root.w += val if (i % 2 == 0) else -val
        return
    
    def build_tree(self):
        """
        num_nodes is the number of nodes to add after the root, first iteration is to get root's prior, that's why we do num_nodes + 1 iterations
        Note: the root will be the first node returned by self.select(), which we want to fwd pass to get prior over its children (with dirichlet noise added)
              but don't want to backprop its value to itself otherwise root.N = sum(root.children.n) + 1

        Note: we evaluate node with torch.no_grad() since these tensors will only be used to build tree/dataset so don't need its gradients

        We use add_noise=True to generate self-play training data to the root explores moves the network doesn't like
        We use add_noise=False to play aginst AI
        """
        for _ in range(self.num_nodes + 1):
            node = self._select()

            # if selected node is a terminal node, get stored reward, else evaluate via fwd pass to get reward
            value = None
            if node.state.game_ended:
                value = node.state.reward
            else:
                board_tensor = torch.tensor(node.state.board, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    value, logits = self.model(board_tensor)  # fwd pass returns [1, 1] value and [1, 7] logits
                    value = value.item()

                # mask illegal moves and store action probs in node
                for i in list(set(range(COLS)) - set(node.state.allowed_moves)):
                    logits[0,i] = -np.inf
                node.prior = F.softmax(logits, dim=1)  # [1, 7]
                node.evaluated = True

                # for root node, add dirichlet noise to its prior so it explores actions the network doesn't like
                if node.id == self.root.id and self.add_noise:
                    # generate symmetric dirichlet distribution with parameter alpha for all 7 moves
                    alphas = [ALPHA] * COLS
                    rng = np.random.default_rng()
                    noise = rng.dirichlet(alphas)

                    # redistribute to only the allowed moves it there's some illegal moves
                    if self.root.state.allowed_moves.size != COLS:
                        full_noise = noise.copy()
                        noise = np.zeros(COLS)
                        for a in self.root.state.allowed_moves:
                            noise[a] = full_noise[a]
                        noise = noise / noise.sum()

                    # add noise to root's prior weighted by epsilon
                    self.root.prior = (1-EPS) * self.root.prior + EPS * noise
                    self.root.prior = self.root.prior / self.root.prior.sum()

            # skip backprop for root
            if node.id == self.root.id:
                continue
            self._backpropagate(node, value)  # update stats of current branch

    def choose_move(self, sample=False):
        """
        choose action corresponding to child with greatest visit count
        if there is a tie we return the first child in the order it was added (which is random)
        a tie break could be implemented based on greatest Q but would rarely be needed if building large enough tree
        Note: don't assume root will have 7 children as later in game it won't have

        We use sample=True to generate self-play training data to generate diverse set of games with diverse opennings
        We use sample=False to actually play against AI
        """
        assert len(self.root.children) > 0, "root node must have at least 1 child to choose move from MCTS"
        best_move = 0
        max_score = 0
        probs = np.zeros(COLS)

        # compute argmax and probs of actions
        for a, c in self.root.children:
            probs[a] = c.n / self.root.n  # put prob in index corresponding to action
            if c.n > max_score:
                max_score = c.n
                best_move = a

        # change to sampling if parameter is true. By default we do argmax
        if sample:
            rng = np.random.default_rng()
            best_move = rng.choice(len(probs), p=probs)  # sample index corresponding to action (will ignore illegal moves with 0 prob)
            
        return best_move
        