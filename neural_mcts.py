import numpy as np
import torch
import torch.nn.functional as F
from game_interface import State, Node

FPU = 0
ALPHA = 0.3
EPS = 0.25
C = 1.5 # conventional value of 2 * np.sqrt(2) for reward [-1, 1] puts too much weight on exploration


class NeuralMCTS:
    def __init__(self, root: Node, model, num_nodes, add_noise=False):
        assert root.id == 0, "root node of MCTS must have id = 0"
        self.root = root
        self.cols = root.state.cols
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
        node = self.root

        # go thorugh evaluated nodes (have stored prior for children) that are non-terminal (haven't ended game)
        while node.evaluated and not node.state.game_ended:            

            # calculate PUCT (Polynomial Upper Confidence bounds applied to Trees) criteria
            # start at -inf so illegal moves won't be updated and hence won't be chosen in the argmax
            PUCT_criteria = np.full(self.cols, -np.inf)
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
                for i in list(set(range(self.cols)) - set(node.state.allowed_moves)):
                    logits[0,i] = -np.inf
                node.prior = F.softmax(logits, dim=1)  # [1, 7]
                node.evaluated = True

                # for root node, add dirichlet noise to its prior so it explores actions the network doesn't like
                if node.id == self.root.id and self.add_noise:
                    # generate symmetric dirichlet distribution with parameter alpha for all 7 moves
                    alphas = [ALPHA] * self.cols
                    rng = np.random.default_rng()
                    noise = rng.dirichlet(alphas)

                    # redistribute to only the allowed moves it there's some illegal moves
                    if self.root.state.allowed_moves.size != self.cols:
                        full_noise = noise.copy()
                        noise = np.zeros(self.cols)
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
        best_move = -1
        max_score = 0
        probs = np.zeros(self.cols)

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