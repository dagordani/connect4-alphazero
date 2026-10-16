// Search tree replay and layout.
//
// The engine sends `analysis.tree`: one event per backpropagation, in order,
// [nodeId, parentId, action, value] (parentId/action = -1 when the node already
// existed). TreeReplay applies them one at a time with the same update as
// _backpropagate in the MCTS classes, so after all events every node's n and w
// match the engine's tree exactly.

export class TreeReplay {
  constructor(events) {
    this.events = events;
    this.reset();
  }

  reset() {
    this.root = { id: 0, parent: null, action: null, n: 0, w: 0, children: [] };
    this.nodes = new Map([[0, this.root]]);
    this.applied = 0;
    this.lastId = null; // node of the most recent event, to highlight its path
  }

  // Apply events until `count` have been applied (rewinding by replaying from scratch).
  advanceTo(count) {
    const target = Math.max(0, Math.min(count, this.events.length));
    if (target < this.applied) this.reset();
    while (this.applied < target) this.#apply(this.events[this.applied++]);
  }

  #apply([id, parentId, action, value]) {
    let node = this.nodes.get(id);
    if (!node) {
      const parent = this.nodes.get(parentId);
      node = { id, parent, action, n: 0, w: 0, children: [] };
      parent.children.push(node);
      this.nodes.set(id, node);
    }
    // same walk as _backpropagate: +value at the node, alternating sign per level up
    let sign = 1;
    for (let cur = node; cur; cur = cur.parent) {
      cur.n += 1;
      cur.w += sign * value;
      sign = -sign;
    }
    this.lastId = id;
  }

  get done() {
    return this.applied === this.events.length;
  }
}

// Port of viz.py tree_layout: every drawn leaf gets its own x slot (0, 1, 2, ...
// left to right), every parent sits at the midpoint of its first and last child,
// children ordered by column. Nodes with fewer than `minVisits` visits are left
// out with their whole subtree, the only way to keep a big tree readable.
export function treeLayout(root, minVisits = 1, maxDepth = null) {
  const pos = new Map(); // id -> {x, depth}
  const edges = []; // [parent, child]
  let nextX = 0;
  let deepest = 0;

  function place(node, depth) {
    deepest = Math.max(deepest, depth);
    let kids = node.children.filter((c) => c.n >= minVisits).sort((a, b) => a.action - b.action);
    if (maxDepth != null && depth >= maxDepth) kids = [];
    let x;
    if (!kids.length) {
      x = nextX++;
    } else {
      const xs = kids.map((c) => place(c, depth + 1));
      x = (xs[0] + xs[xs.length - 1]) / 2;
      for (const c of kids) edges.push([node, c]);
    }
    pos.set(node.id, { x, depth });
    return x;
  }

  place(root, 0);
  return { pos, edges, leaves: nextX, depth: deepest };
}

function countLeaves(node, minVisits) {
  const kids = node.children.filter((c) => c.n >= minVisits);
  return kids.length ? kids.reduce((sum, c) => sum + countLeaves(c, minVisits), 0) : 1;
}

// Smallest visit threshold that keeps the drawn tree to `maxLeaves` leaves. The leaf
// count only shrinks as the threshold grows, so binary search it.
export function autoMinVisits(root, maxLeaves) {
  let lo = 1;
  let hi = Math.max(1, root.n);
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (countLeaves(root, mid) <= maxLeaves) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}
