// Port of neural_mcts.py (NeuralMCTS), without the Dirichlet noise that only
// self-play training uses. Keep the constants and the selection rule in sync with
// the Python class; scripts/parity.mjs compares the two on the same positions.
//
// `evaluate(state)` runs the network: async, returns {value, logits} for the side to move.
// Every _backpropagate call is also logged in `events` for the UI's tree replay:
// [nodeId, parentId, action, value], parentId/action -1 when the node already existed
// (same format as engine.py's Recording wrapper).

import { argmax, backpropagate, Node, sampleIndex } from './game.js';

const FPU = 0;
const C = 1.5; // conventional value of 2 * sqrt(2) for reward [-1, 1] puts too much weight on exploration

// Masked softmax in float32 like torch's F.softmax on float32 logits, so priors (and
// therefore PUCT ties) match the Python search as closely as possible.
export function maskedSoftmax(logits, allowedMoves) {
  const f = Math.fround;
  const allowed = new Set(allowedMoves);
  let max = -Infinity;
  for (const a of allowed) max = Math.max(max, logits[a]);
  const exps = Array.from(logits, (l, a) => (allowed.has(a) ? f(Math.exp(f(l - max))) : 0));
  let sum = 0;
  for (const e of exps) sum = f(sum + e);
  return exps.map((e) => f(e / sum));
}

export class NeuralMCTS {
  constructor(root, evaluate, numNodes) {
    if (root.id !== 0) throw new Error('root node of MCTS must have id = 0');
    this.root = root;
    this.cols = root.state.cols;
    this.evaluate = evaluate;
    this.numNodes = numNodes;
    this.size = 1;
    this.events = [];
    this.seen = new Set([root.id]);
  }

  // PUCT: Q(a) + C * P(a) * sqrt(N) / (1 + n(a)); returns the first unevaluated or terminal node
  _select() {
    let node = this.root;

    // go through evaluated nodes (have stored prior for children) that are non-terminal
    while (node.evaluated && !node.state.gameEnded) {
      // start at -inf so illegal moves won't be chosen in the argmax
      const criteria = new Array(this.cols).fill(-Infinity);
      for (const a of node.state.allowedMoves) {
        // if child is in graph, use its stats, else use prior
        const p = node.prior[a];
        const entry = node.children.find(([i]) => i === a);
        if (entry) {
          const child = entry[1];
          const exploitation = -1 * (child.w / child.n);
          const exploration = (p * Math.sqrt(node.n)) / (1 + child.n);
          criteria[a] = exploitation + C * exploration;
        } else {
          // untried moves differ only in the prior; root.n is 0 on the first pass, hence max(n, 1)
          criteria[a] = FPU + C * p * Math.sqrt(Math.max(node.n, 1));
        }
      }

      // choose best action, adding the child to the graph if not present
      const action = argmax(criteria);
      let child = node.children.find(([a]) => a === action)?.[1];
      if (!child) {
        child = new Node(this.size, node.state.transitionState(action));
        node.addChild(action, child);
        this.size += 1;
      }
      node = child;
    }
    return node;
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

  // numNodes + 1 iterations: the first one only evaluates the root (no backprop on the root,
  // otherwise root.n = sum(children n) + 1)
  async buildTree() {
    for (let k = 0; k < this.numNodes + 1; k++) {
      const node = this._select();

      // terminal node: stored reward; otherwise evaluate with the network
      let value;
      if (node.state.gameEnded) {
        value = node.state.reward;
      } else {
        const out = await this.evaluate(node.state);
        value = out.value;
        node.prior = maskedSoftmax(out.logits, node.state.allowedMoves);
        node.evaluated = true;
      }

      if (node.id === this.root.id) continue; // skip backprop for root
      this._backpropagate(node, value);
    }
  }

  // most visited child; ties go to the first child added. sample=true draws in proportion to visits.
  chooseMove(sample = false, random = Math.random) {
    if (!this.root.children.length) throw new Error('root node must have at least 1 child to choose move from MCTS');
    let bestMove = -1;
    let maxScore = 0;
    const probs = new Array(this.cols).fill(0);
    for (const [a, c] of this.root.children) {
      probs[a] = c.n / this.root.n;
      if (c.n > maxScore) {
        maxScore = c.n;
        bestMove = a;
      }
    }
    if (sample) bestMove = sampleIndex(probs, random);
    return bestMove;
  }
}
