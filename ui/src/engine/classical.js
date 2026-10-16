// Port of classical_mcts.py (ClassicalMCTS): UCT selection, random expansion, random
// rollouts, no network. Keep in sync with the Python class. Tree events are logged
// like in neural.js for the UI's replay.

import { argmax, backpropagate, Node } from './game.js';

const C = 2 * Math.sqrt(2);

export class ClassicalMCTS {
  constructor(root, numNodes, maxSimPlies, random = Math.random) {
    if (root.id !== 0) throw new Error('root node of MCTS must have id = 0');
    this.root = root;
    this.cols = root.state.cols;
    this.numNodes = numNodes;
    this.maxSimPlies = maxSimPlies;
    this.random = random;
    this.size = 1;
    this.events = [];
    this.seen = new Set([root.id]);
  }

  // Follow the best UCT child while the node has all its children in the tree.
  // Q = W/N negated (the child's value is for the opponent); exploration sqrt(ln N / n).
  _select() {
    let node = this.root;
    while (node.fullyExpanded) {
      const criteria = new Array(this.cols).fill(-Infinity);
      for (const [a, c] of node.children) {
        const exploitation = -1 * (c.w / c.n);
        const exploration = Math.sqrt(Math.log(node.n) / c.n);
        criteria[a] = exploitation + C * exploration;
      }
      const action = argmax(criteria);
      node = node.children.find(([a]) => a === action)[1];
    }
    return node;
  }

  // add a child of `node` for a random action not yet in the tree
  _expand(node) {
    const taken = new Set(node.children.map(([a]) => a));
    const possible = node.state.allowedMoves.filter((a) => !taken.has(a));

    // if only 1 choice left, the node becomes fully expanded
    let action;
    if (possible.length === 1) {
      action = possible[0];
      node.fullyExpanded = true;
    } else {
      action = possible[Math.floor(this.random() * possible.length)];
    }

    // a game-ending move stores its reward in the child already
    const child = new Node(this.size, node.state.transitionState(action));
    node.addChild(action, child);
    child.fullyExpanded = false; // even with no allowed moves: terminal nodes skip simulate()
    this.size += 1;
    return child;
  }

  // random moves from `state` (modified in place) until the game ends or maxPlies;
  // the value is for the player to move at the start
  _simulate(state, maxPlies) {
    let plies = 0;
    while (plies < maxPlies && !state.gameEnded) {
      const moves = state.allowedMoves;
      state.transitionStateInPlace(moves[Math.floor(this.random() * moves.length)]);
      plies += 1;
    }
    // game over: its reward; cut off: a random value in [-1, 1]
    let value = state.gameEnded ? state.reward : this.random() * 2 - 1;
    // the reward is stored for the player to move at the end; negate when that is the opponent
    if (plies % 2 === 1) value *= -1;
    return value;
  }

  _backpropagate(node, val) {
    backpropagate(this.root, node, val);
    if (this.seen.has(node.id)) {
      this.events.push([node.id, -1, -1, val]);
    } else {
      this.seen.add(node.id);
      const action = node.parent.children.find(([, c]) => c === node)[0];
      this.events.push([node.id, node.parent.id, action, val]);
    }
  }

  // numNodes iterations, each adding one node (except when a terminal node is selected again)
  buildTree() {
    for (let k = 0; k < this.numNodes; k++) {
      const parent = this._select();
      const child = parent.state.gameEnded ? parent : this._expand(parent);
      const value = child.state.gameEnded ? child.state.reward : this._simulate(child.state.copyState(), this.maxSimPlies);
      this._backpropagate(child, value);
    }
  }

  // most visited child; ties go to the first child added
  chooseMove() {
    if (!this.root.children.length) throw new Error('root node must have at least 1 child to choose move from MCTS');
    let bestMove = -1;
    let maxScore = 0;
    for (const [a, c] of this.root.children) {
      if (c.n > maxScore) {
        maxScore = c.n;
        bestMove = a;
      }
    }
    return bestMove;
  }
}
