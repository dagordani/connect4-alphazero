import numpy as np
from game_interface import State, Node, Game

C = 2 * np.sqrt(2)


class ClassicalMCTS:
    def __init__(self, root: Node, num_nodes: int, max_sim_plies: int):
        assert root.id == 0, "root node of MCTS must have id = 0"
        self.root = root
        self.cols = root.state.cols
        self.num_nodes = num_nodes
        self.max_sim_plies = max_sim_plies
        self.size = 1

    def _select(self) -> Node:
        """
        find first node that is not fully expanded yet, selecting the most promising path at each level
        exploitation: Q = W/N = total reward / amount of times this move was tried = "win rate". Negate since parent wants weakest state of child since its opponent
        exploration: higher for children less visited relative to siblings. Use ln(x) to scale slowly and sqrt(x) to get unbiased statistic as num_simulations->inf
        """
        node = self.root

        # when current node has all its children in graph we choose the best one. Otherwise, return current node
        # we skip fully expanded nodes or nodes with no allowed moves (can't have children)
        # can't set nodes that can't have children as .fully_expanded=True since we would enter while loop and get index error at node.children[best_node_idx]
        while node.fully_expanded:

            UCT_criteria = np.full(self.cols, -np.inf)
            for (a, c) in node.children:

                # calculate UCT (Upper Confidence bounds applied to Trees) criteria
                exploitation = -1 * (c.w / c.n) # Q = W/N = "win rate"
                exploration = np.sqrt(np.log(node.n) / c.n)
                criteria = exploitation + C * exploration
                UCT_criteria[a] = criteria

            # choose best child
            node = node.children[UCT_criteria.argmax()]

        return node

    def _expand(self, node: Node) -> Node:
        """
        add child of 'node' for random action not yet in graph
        """
        assert node.fully_expanded == False, "expand() cannot be called in a fully expanded node"
        assert node.state.allowed_moves.size != 0, "expand() cannot be called in a node with no allowed moves"
        
        # get all possible actions that have not been added to tree
        legal_actions = set(node.state.allowed_moves)
        actions_taken = set()
        
        for a, c in node.children:
            actions_taken.add(a)
        possible_actions = legal_actions - actions_taken

        # if only 1 choice left, set node as fully_expanded. Otherwise, choose random action
        action = None
        if len(possible_actions) == 1:
            action = list(possible_actions)[0]
            node.fully_expanded = True
        else:
            action = np.random.choice(np.array(list(possible_actions)))

        # create new state based on chosen action
        # if it's a game ending move, the reward gets saved for the child already
        # reason its the interface works better for simulate() than generate(), there we have only 1 state, here we have 2
        # parent state and child state, we make action from copy of parent state that becomes child state so it gets the reward
        child_state = node.state.transition_state(action)

        # create child Node
        child = Node(self.size, child_state)

        # add child to graph
        node.add_child(action, child)
        child.fully_expanded = False  # even if no allowed moves as that would be a terminal node and we skip those from simulate()
        self.size += 1
        
        return child

    def _simulate(self, g: Game, max_plies: int) -> float:
        """
        starting from a given state (in Game object), play random actions until end of game to get reward
        """
        # value function
        rng = np.random.default_rng()
        def model():
            return rng.uniform(-1, 1)
        
        plies = 0
        value = None
        while plies < max_plies and not g.state.game_ended:
            # make random move
            action = int(np.random.choice(g.state.allowed_moves))
            g.move(action)
            plies += 1
        
        # check if game reached terminal state to get win/draw/loose reward, otherwise call value function

        # reward is always stored for the player that is about to move (the state after the winning move)
        # and we always want to assign the reward to the player that starts the simulation
        # when plies is odd, starting player made last move so opponent is to move so the reward is of opponent, so we negate
        value = g.state.reward if g.state.game_ended else model()
        if plies % 2 == 1:
            value *= -1

        return value
        
    def _backpropagate(self, node: Node, val: int) -> None:
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
    
    def build_tree(self) -> None:
        """
        num_nodes is the number of nodes to add to graph after root node. This is also the number of iterations since we add one node each iteration
        """
        for _ in range(self.num_nodes):
            parent = self._select()
                
            # if selected node is a terminal node we skip expand() since there's no children to add and we already have .reward
            # Note: in a previous iteration this terminal node was generated by expand()
            if parent.state.game_ended:
                child = parent
            else:
                child = self._expand(parent)

            # if expand() generates a terminal node skip simulate() since we already have .reward
            # else, simulate a game to get reward of child node and backpropagate
            value = None
            if child.state.game_ended:
                value = child.state.reward
            else:
                game = Game(child.state.copy_state())
                value = self._simulate(g=game, max_plies=self.max_sim_plies)  # calculate value of position            
            self._backpropagate(child, value)  # update stats of current branch

    def choose_move(self) -> int:
        """
        choose move based on greatest visit count N. If there is a tie we pick the first one based on order it was added
        Note:
        - we could implement tie break based on Q but with large number of nodes it should happen less and less
        - we can also allow sampling instead of always argmax since the visit counts form a probabiliy distribution
        """
        assert len(self.root.children) > 0, "root node must have at least 1 child to choose move from MCTS"
        best_move = -1
        max_score = 0
        for (a, c) in self.root.children:
            if c.n > max_score:
                max_score = c.n
                best_move = a
        return best_move