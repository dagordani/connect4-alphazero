// What the engine thought about the position: values plus prior / visits / Q per column,
// laid out horizontally (one table column per board column). While the tree animates,
// N, Q and the search value follow the replay.
// Classical MCTS has no network, so its analysis has no priors or network value.

const signed = (x) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${x.toFixed(3)}`);

export default function SearchPanel({ analysis }) {
  if (!analysis) {
    return (
      <section className="panel stats">
        <h2>Search</h2>
        <p className="meta">Engine moves and hints show their analysis here.</p>
      </section>
    );
  }

  const searched = analysis.simulations > 0;
  const neural = analysis.opponent !== 'classical';
  const moves = [...analysis.moves].sort((a, b) => a.move - b.move);

  return (
    <section className="panel stats">
      <h2>Search</h2>
      <div className="stats-body">
        <div className="values">
          <div>
            <div className="value">{signed(analysis.search_value)}</div>
            <div className="meta">search value</div>
          </div>
          {neural && (
            <div>
              <div className="value">{signed(analysis.net_value)}</div>
              <div className="meta">network value</div>
            </div>
          )}
        </div>

        <div className="stats-scroll">
          <table className="stats-table">
            <thead>
              <tr>
                <th>col</th>
                {moves.map((m) => (
                  <th key={m.move}>{m.move}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {neural && (
                <tr>
                  <th title="network prior">P</th>
                  {moves.map((m) => (
                    <td key={m.move}>{m.prior.toFixed(3)}</td>
                  ))}
                </tr>
              )}
              <tr>
                <th title="visit count">N</th>
                {moves.map((m) => (
                  <td key={m.move}>{m.visits ?? '—'}</td>
                ))}
              </tr>
              <tr>
                <th title="value of the move for the side to move">Q</th>
                {moves.map((m) => (
                  <td key={m.move}>{signed(m.q)}</td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>

        <p className="note">
          For the side to move, −1 lost … +1 won ·{' '}
          {searched ? `${analysis.simulations} simulations` : 'no search (policy only)'}
          {analysis.sampled && ' · move was sampled'}
          {!neural && ' · classical MCTS'}
          <br />
          {neural ? (
            <>
              <b>P</b> network prior, <b>N</b> visits, <b>Q</b> value found for the move. When N disagrees with P,
              the search is correcting the network.
            </>
          ) : (
            <>
              <b>N</b> visits, <b>Q</b> average result of the random games through the move. With no network, it
              needs far more simulations to play well.
            </>
          )}
        </p>
      </div>
    </section>
  );
}
