import { useEffect, useLayoutEffect, useRef, useState } from 'react';

import { autoMinVisits, treeLayout } from '../tree.js';

// The search tree as it is built, drawn like viz.py print_mcts_tree: one circle per
// node with its id, visit count N and average value Q = W/N (for the player to move
// at that node), darker = more visits, edges labelled with the column played.
// Big trees are cropped to the most-visited nodes (as many leaves as fit the panel
// width) unless "Full tree" is on; the panel scrolls, zooms (buttons or ctrl/⌘ +
// wheel) and pans by dragging. "Fit" fits the width; deep trees scroll vertically.

const SLOT = 58; // px per leaf
const LEVEL = 86; // px per depth level
const R = 23; // node radius
const PAD = 34;
const CROP_SLOT = SLOT * 0.85; // cropped trees keep at least this much width per leaf
const ZOOM_MIN = 0.05;
const ZOOM_MAX = 3;

const signed = (x, digits = 2) => `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`;

export default function TreeView({ analysis, tree, step, animating, busy, onSkip }) {
  let body;
  if (!analysis) {
    body = (
      <div className="tree-empty">
        {busy ? 'The engine is thinking…' : 'The search tree is drawn here when the engine moves or you ask for a hint.'}
      </div>
    );
  } else if (!analysis.tree) {
    body = <NetworkView analysis={analysis} />;
  }

  return (
    <section className="panel tree-panel">
      {body ?? <SearchTree analysis={analysis} tree={tree} step={step} animating={animating} onSkip={onSkip} />}
    </section>
  );
}

function SearchTree({ analysis, tree, step, animating, onSkip }) {
  const [full, setFull] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [fit, setFit] = useState(true); // zoom follows the panel width until the user zooms
  const scrollRef = useRef(null);
  const centerRoot = useRef(false);
  const size = useElementSize(scrollRef);

  const cropLeaves = size.width ? Math.max(6, Math.floor((size.width - 2 * PAD) / CROP_SLOT)) : 16;
  const minVisits = full ? 1 : autoMinVisits(tree.root, cropLeaves);
  const { pos, edges, leaves, depth } = treeLayout(tree.root, minVisits);
  const width = Math.max(1, leaves) * SLOT + 2 * PAD;
  const height = depth * LEVEL + 2 * PAD;
  const fitZoom = size.width ? clamp(Math.min((size.width - 2) / width, 1)) : 1;
  const scale = fit ? fitZoom : zoom;

  // A new search or switching to the cropped tree fits the width; the full tree
  // opens at 100% (readable) scrolled to the root, to be explored by panning.
  useEffect(() => {
    if (full) {
      setFit(false);
      setZoom(1);
      centerRoot.current = true;
    } else {
      setFit(true);
    }
  }, [analysis, full]);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!centerRoot.current || !el) return;
    centerRoot.current = false;
    el.scrollLeft = (PAD + (pos.get(0).x + 0.5) * SLOT) * scale - el.clientWidth / 2;
    el.scrollTop = 0;
  });

  function zoomTo(next) {
    setFit(false);
    setZoom(clamp(next));
  }

  // ctrl/⌘ + wheel zooms (needs a non-passive listener to stop the page zooming)
  useEffect(() => {
    const el = scrollRef.current;
    const onWheel = (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      setFit(false);
      setZoom((z) => clamp((fit ? fitZoom : z) * Math.exp(-e.deltaY * 0.002)));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [fit, fitZoom]);

  const drag = useRef(null);
  const onPointerDown = (e) => {
    const el = scrollRef.current;
    drag.current = { x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop };
    el.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const el = scrollRef.current;
    el.scrollLeft = drag.current.left - (e.clientX - drag.current.x);
    el.scrollTop = drag.current.top - (e.clientY - drag.current.y);
  };
  const onPointerUp = () => (drag.current = null);

  // nodes on the path of the latest backpropagation, highlighted while animating
  const path = new Set();
  if (animating && tree.lastId != null) {
    for (let n = tree.nodes.get(tree.lastId); n; n = n.parent) path.add(n.id);
  }

  const maxN = Math.max(1, ...edges.map(([, c]) => c.n));
  const at = (id) => {
    const p = pos.get(id);
    return [PAD + (p.x + 0.5) * SLOT, PAD + p.depth * LEVEL];
  };
  const total = tree.nodes.size;
  const kind = analysis.opponent === 'classical' ? 'Classical MCTS' : 'Neural MCTS';

  return (
    <>
      <div className="tree-head">
        <h2>Search tree</h2>
        <span className="meta">
          {kind} · {step} / {analysis.tree.length} simulations · {total} nodes
          {pos.size < total && ` · showing ${pos.size} with N ≥ ${minVisits}`}
        </span>
        <span className="tree-tools">
          {animating && <button onClick={onSkip}>Skip</button>}
          <label className="check">
            <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} />
            Full tree
          </label>
          <span className="segmented">
            <button onClick={() => zoomTo(scale / 1.25)} title="Zoom out">−</button>
            <button onClick={() => setFit(true)} className={fit ? 'on' : ''} title="Fit to panel">
              Fit
            </button>
            <button onClick={() => zoomTo(scale * 1.25)} title="Zoom in">+</button>
          </span>
        </span>
      </div>

      <div
        ref={scrollRef}
        className="tree-scroll"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <svg width={width * scale} height={height * scale} viewBox={`0 0 ${width} ${height}`}>
          {edges.map(([p, c]) => {
            const [x1, y1] = at(p.id);
            const [x2, y2] = at(c.id);
            return (
              <g key={c.id}>
                <line className={`edge${path.has(c.id) ? ' hot' : ''}`} x1={x1} y1={y1 + R} x2={x2} y2={y2 - R} />
                <text className="edge-label" x={(x1 + x2) / 2} y={(y1 + R + y2 - R) / 2 + 4}>
                  {c.action}
                </text>
              </g>
            );
          })}
          {[...pos.keys()].map((id) => {
            const node = tree.nodes.get(id);
            const [x, y] = at(id);
            const shade = Math.min(1, node.n / (maxN * 1.3)); // like the Blues colormap in viz.py
            const q = node.n ? node.w / node.n : 0;
            return (
              <g key={id} className={`node${path.has(id) ? ' hot' : ''}${shade > 0.5 ? ' dark' : ''}`}>
                <title>{`node #${id}${id === 0 ? ' (root)' : `, column ${node.action}`}\nN = ${node.n}\nQ = ${signed(q, 3)} for the player to move here`}</title>
                <circle
                  cx={x}
                  cy={y}
                  r={R}
                  style={{ fill: `color-mix(in srgb, var(--accent) ${Math.round(8 + 82 * shade)}%, var(--panel))` }}
                />
                <text x={x} y={y - 8}>{id === 0 ? 'root' : `#${id}`}</text>
                <text x={x} y={y + 3}>N={node.n}</text>
                <text x={x} y={y + 14}>{signed(q)}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="meta">
        Edge = column played · Q = W/N for the player to move at that node · darker = more visits · drag to pan,
        ctrl/⌘ + scroll to zoom
      </div>
    </>
  );
}

// simulations = 0: no tree, so show what the network itself returned.
function NetworkView({ analysis }) {
  const byMove = new Map(analysis.moves.map((m) => [m.move, m]));
  const cols = Math.max(...analysis.moves.map((m) => m.move)) + 1;
  const maxP = Math.max(...analysis.moves.map((m) => m.prior));
  const v = analysis.net_value;

  return (
    <>
      <div className="tree-head">
        <h2>Network output</h2>
        <span className="meta">0 simulations: no search, the policy head picks the move</span>
      </div>
      <div className="net-view">
        <div className="net-value">
          <h3 className="head-title">Value Head</h3>
          <div className="value">{signed(v, 3)}</div>
          <div className="meta">value for the side to move</div>
          <div className="meter">
            <span className="meter-mid" />
            <span className="meter-mark" style={{ left: `${50 + 50 * v}%` }} />
          </div>
          <div className="meter-scale">
            <span>−1 lost</span>
            <span>0</span>
            <span>+1 won</span>
          </div>
        </div>

        <div className="policy">
          <h3 className="head-title">Policy Head</h3>
          <div className="meta">probability of each column (illegal moves masked)</div>
          <div className="policy-bars">
            {Array.from({ length: cols }, (_, c) => {
              const m = byMove.get(c);
              return (
                <div key={c} className="policy-col">
                  <div className="policy-track">
                    {m && (
                      <div
                        className={`policy-bar${m.prior === maxP ? ' top' : ''}`}
                        style={{ height: `${(100 * m.prior) / maxP}%` }} // tallest bar = most likely column
                      >
                        <span>{(100 * m.prior).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                  <div className="policy-label">{c}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}

function clamp(z) {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
}

function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}
