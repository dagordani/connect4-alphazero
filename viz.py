import networkx as nx
import matplotlib.pyplot as plt

def tree_layout(root_node, min_visits=1, max_depth=None):
    """Top-down positions where nodes can never overlap.

    Every *leaf* of the drawn tree gets its own x slot (0, 1, 2, ... in
    left-to-right order), and every parent sits at the midpoint of its first and
    last child. A wide subtree therefore takes as much room as it needs, instead
    of each child getting an equal share of its parent's width.

    Nodes with fewer than `min_visits` visits, or deeper than `max_depth`, are
    left out (with their whole subtree) -- the only way to keep a big tree readable.
    """
    pos, edges = {}, []
    next_x = [0]

    def place(node, depth):
        kids = sorted(node.children, key=lambda ac: ac[0])  # left-to-right by column
        kids = [(a, c) for a, c in kids if c.n >= min_visits]
        if max_depth is not None and depth >= max_depth:
            kids = []
        if not kids:
            x = next_x[0]
            next_x[0] += 1
        else:
            xs = [place(c, depth + 1) for _, c in kids]
            x = (xs[0] + xs[-1]) / 2
            edges.extend((node.id, c.id, a) for a, c in kids)
        pos[node.id] = (x, -depth)
        return x

    place(root_node, 0)
    return pos, edges, next_x[0]

def print_mcts_tree(root_node, min_visits=1, max_depth=None, slot_inches=0.9, max_width_inches=60):
    """Draw the search tree: one circle per node, edges labelled with the column played.

    Each node shows its id, visit count N and average value Q/N (from the view
    of the player to move at that node). Darker = more visits.
    """
    pos, edges, n_leaves = tree_layout(root_node, min_visits, max_depth)
    depth = int(-min(y for _, y in pos.values()))

    G = nx.DiGraph()
    nodes = {root_node.id: root_node}
    stack = [root_node]
    while stack:
        nd = stack.pop()
        for _, c in nd.children:
            nodes[c.id] = c
            stack.append(c)
    G.add_nodes_from(pos)
    G.add_edges_from((u, v) for u, v, _ in edges)

    labels = {i: f"#{i}\nN={nodes[i].n}\n{nodes[i].w / max(nodes[i].n, 1):+.2f}" for i in pos}
    visits = [nodes[i].n for i in G.nodes]

    # One leaf slot = `slot_inches` wide, so the figure grows with the tree
    # instead of squeezing it. Very large trees hit max_width_inches: prune them
    # with min_visits / max_depth rather than shrinking the nodes.
    width = min(max(6, n_leaves * slot_inches), max_width_inches)
    fig, ax = plt.subplots(figsize=(width, 1.5 * (depth + 1)))
    if n_leaves * slot_inches > max_width_inches:
        print(f"{n_leaves} leaves don't fit in {max_width_inches} in -- "
              f"try min_visits={min_visits * 2} or max_depth")

    # Node size scales with the slot width actually used (points^2).
    slot_pts = 72 * width / max(n_leaves, 1)
    node_size = min(2200, (0.8 * slot_pts) ** 2)
    font = max(4, min(8, slot_pts / 8))

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edge_color="#999", width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=visits,
                           cmap="Blues", vmin=0, vmax=max(visits) * 1.3,
                           edgecolors="#4a6fa5", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=font)
    nx.draw_networkx_edge_labels(G, pos, {(u, v): str(a) for u, v, a in edges}, ax=ax,
                                 font_size=font, font_color="#c0503f", rotate=False,
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    ax.set_title(f"MCTS tree: {len(pos)} of {len(nodes)} nodes shown"
                 + (f" (N >= {min_visits})" if min_visits > 1 else "")
                 + (f" (depth <= {max_depth})" if max_depth is not None else "")
                 + "   edge = column played, Q/N for the player to move")
    ax.margins(x=0.5 / max(n_leaves, 1), y=0.15)
    ax.axis("off")
    plt.tight_layout()
    plt.show()