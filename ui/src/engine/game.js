// Port of game_interface.py (State, Node). Keep the two in sync: the parity test
// (scripts/parity.mjs) plays random games through both and compares every position.
//
// The board is stored in the side-to-move frame: +1 = player to move, -1 = opponent,
// flipped after every move. board[r * cols + c], row 0 = top.

export class State {
  constructor(rows, cols, board, lastMoveCoordinates = null, gameEnded = false, reward = null) {
    this.rows = rows;
    this.cols = cols;
    this.board = board;
    this.allowedMoves = [];
    for (let c = 0; c < cols; c++) if (board[c] === 0) this.allowedMoves.push(c);
    this.skyline = new Array(cols).fill(0);
    for (let i = 0; i < board.length; i++) if (board[i] !== 0) this.skyline[i % cols] += 1;
    this.lastMoveCoordinates = lastMoveCoordinates;
    this.gameEnded = gameEnded;
    this.reward = reward;
  }

  static fromEmptyBoard(rows, cols) {
    return new State(rows, cols, new Int8Array(rows * cols));
  }

  copyState() {
    return new State(this.rows, this.cols, this.board.slice(), this.lastMoveCoordinates, this.gameEnded, this.reward);
  }

  _updateBoard(action) {
    const x = this.rows - this.skyline[action] - 1;
    const y = action;
    this.lastMoveCoordinates = [x, y];
    this.board[x * this.cols + y] = 1; // update board
    this.skyline[y] += 1; // update skyline
    this.allowedMoves = this.allowedMoves.filter((c) => this.board[c] === 0); // update allowed moves
  }

  // used once in expand(), many times in simulate()
  transitionStateInPlace(action) {
    this._updateBoard(action); // make move
    this._checkWinningCondition(); // update gameEnded and reward
    // flip board to pass to other player. Call after _checkWinningCondition since that checks for +1 pieces
    for (let i = 0; i < this.board.length; i++) this.board[i] = -this.board[i];
  }

  transitionState(action) {
    const state = new State(this.rows, this.cols, this.board.slice()); // lastMoveCoordinates=null, gameEnded=false
    state.transitionStateInPlace(action);
    return state;
  }

  // Did the latest move end the game? Win: gameEnded with reward -1 (the player to move
  // next lost). Draw: board full, reward 0. Same sliding windows of 4 as the Python code.
  _checkWinningCondition() {
    const CONN = 4;
    const DIRS = [
      [0, 1],
      [-1, 0],
      [-1, 1],
      [1, 1],
    ]; // right, up, up-right, down-right
    const [x, y] = this.lastMoveCoordinates;
    const xBound = this.rows;
    const yBound = this.cols;
    for (const [dx, dy] of DIRS) {
      for (let d = -CONN + 1; d < 1; d++) {
        // move in negative direction to set starting point of window
        const startX = x + d * dx;
        const startY = y + d * dy;

        // skip out of bounds
        if (startX < 0 || startX >= xBound || startY < 0) continue;

        // move in positive direction to traverse window
        let count = 0;
        for (let i = 0; i < CONN; i++) {
          const xx = startX + i * dx;
          const yy = startY + i * dy;

          // move within bounds
          if (xx >= 0 && xx < xBound && yy >= 0 && yy < yBound) count += this.board[xx * this.cols + yy] === 1 ? 1 : 0;

          if (count === CONN) {
            // player that made last move won, but since the resulting state is of the opponent we set reward to -1
            this.gameEnded = true;
            this.reward = -1;
            return;
          }
        }
      }
    }

    // board is full, and win wasn't found, therefore it's a draw
    if (this.allowedMoves.length === 0) {
      this.gameEnded = true;
      this.reward = 0;
    }
  }
}

export class Node {
  constructor(nodeId, state) {
    this.id = nodeId; // order of nodes created
    this.state = state;
    this.n = 0; // visit count
    this.w = 0; // cumulative reward
    this.parent = null;
    this.children = []; // [action, Node] pairs, in the order they were added
    this.fullyExpanded = false; // for classical MCTS
    this.prior = null; // action probs over its children assigned by policy head
    this.evaluated = false; // whether node has been through the network's forward pass
  }

  addChild(action, child) {
    this.children.push([action, child]);
    child.parent = this;
    child.children = [];
  }
}

// Shared by both searches: increment n and add val with alternating sign from `node`
// up to the root ('val' is for the player to move at `node`). Port of _backpropagate.
export function backpropagate(root, node, val) {
  let cur = node;
  let i = 0;
  while (cur.parent !== null) {
    cur.n += 1;
    cur.w += i % 2 === 0 ? val : -val;
    cur = cur.parent;
    i += 1;
  }
  root.n += 1;
  root.w += i % 2 === 0 ? val : -val;
}

// np.argmax: index of the first maximum
export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

// Index drawn in proportion to `weights` (they need not sum to 1).
export function sampleIndex(weights, random = Math.random) {
  const total = weights.reduce((s, x) => s + x, 0);
  let r = random() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r < 0 && weights[i] > 0) return i;
  }
  return weights.findLastIndex((x) => x > 0);
}
