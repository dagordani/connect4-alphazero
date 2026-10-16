import { useState } from 'react';

import { COLOR, topRow, winningCells } from '../game.js';

// The board: a ghost disc over the hovered column, the grid (size from config.toml), column labels,
// and the search's visit counts (or priors, with search off) as bars underneath.
export default function Board({ board, lastMove, animate, canPlay, legal, humanColor, analysis, onPlay }) {
  const [hover, setHover] = useState(null);
  const wins = winningCells(board);
  const lastRow = lastMove == null ? -1 : topRow(board, lastMove);
  const playable = (c) => canPlay && legal.includes(c);
  const cols = board[0].length;

  return (
    <div className="board-wrap">
      <div className="board-row ghost-row">
        {Array.from({ length: cols }, (_, c) => (
          <div key={c} className="cell">
            {hover === c && playable(c) && <div className={`disc ${humanColor} ghost`} />}
          </div>
        ))}
      </div>

      <div className="board" onMouseLeave={() => setHover(null)}>
        {board.map((row, r) =>
          row.map((v, c) => {
            const isLast = r === lastRow && c === lastMove;
            const classes = ['disc', COLOR[v]];
            if (isLast && animate) classes.push('drop');
            if (isLast && !wins.size) classes.push('last');
            if (wins.has(`${r},${c}`)) classes.push('win');
            return (
              <div
                key={`${r}-${c}`}
                className={`cell${playable(c) ? ' playable' : ''}${hover === c && playable(c) ? ' col-hover' : ''}`}
                onMouseEnter={() => setHover(c)}
                onClick={() => playable(c) && onPlay(c)}
              >
                <div className="hole" />
                {v !== 0 && <div className={classes.join(' ')} style={{ '--fall': r + 1 }} />}
              </div>
            );
          }),
        )}
      </div>

      <div className="board-row col-labels">
        {Array.from({ length: cols }, (_, c) => (
          <div key={c}>{c}</div>
        ))}
      </div>

      <VisitBars analysis={analysis} cols={cols} />
    </div>
  );
}

function VisitBars({ analysis, cols }) {
  const byMove = new Map((analysis?.moves ?? []).map((m) => [m.move, m]));
  const searched = analysis && analysis.simulations > 0;
  const weight = (m) => (searched ? m.visits : m.prior);
  const max = Math.max(1e-9, ...[...byMove.values()].map(weight));

  return (
    <div className="board-row visits" title={searched ? 'visits per column' : 'network prior per column'}>
      {Array.from({ length: cols }, (_, c) => {
        const m = byMove.get(c);
        return (
          <div key={c} className="visit-cell">
            {m && (
              <div className="visit-bar" style={{ height: `${(40 * weight(m)) / max}px` }}>
                <span>{searched ? m.visits : m.prior.toFixed(2)}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
