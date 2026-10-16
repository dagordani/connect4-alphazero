// What the engine thought about the position: values plus prior / visits / Q per column.
// Classical MCTS has no network, so its analysis has no priors or network value.

const signed = (x) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${x.toFixed(3)}`);

export default function SearchPanel({ analysis }) {
  const searched = analysis && analysis.simulations > 0;
  const neural = analysis?.opponent !== 'classical';
  const maxVisits = searched ? Math.max(1, ...analysis.moves.map((m) => m.visits)) : 1;

  return (
    <section className="panel">
      <h2>Search</h2>
      {!analysis ? (
        <p className="meta">Engine moves and hints show their analysis here.</p>
      ) : (
        <>
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
          <div className="meta">
            for the side to move, −1 lost … +1 won ·{' '}
            {searched ? `${analysis.simulations} simulations` : 'no search (policy only)'}
            {!neural && ' · classical MCTS'}
            {analysis.sampled && ' · move was sampled'}
          </div>

          <table>
            <thead>
              <tr>
                <th>col</th>
                {neural && <th>P</th>}
                <th>N</th>
                <th>Q</th>
              </tr>
            </thead>
            <tbody>
              {analysis.moves.map((m) => (
                <tr key={m.move}>
                  <td>{m.move}</td>
                  {neural && <td>{m.prior.toFixed(3)}</td>}
                  <td>
                    {m.visits ?? '—'}
                    {searched && <span className="bar" style={{ width: `${(100 * m.visits) / maxVisits}%` }} />}
                  </td>
                  <td>{signed(m.q)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {neural ? (
        <p className="note">
          <b>P</b> is the network's prior before searching, <b>N</b> is where the search spent its visits,{' '}
          <b>Q</b> is the value it found for that move. When N disagrees with P, the search is correcting the
          network. Keep the simulations fixed when comparing levels.
        </p>
      ) : (
        <p className="note">
          <b>N</b> is where the search spent its visits, <b>Q</b> is the average result of the random games
          played through that move. With no network to guide it, it needs far more simulations to play well.
        </p>
      )}
    </section>
  );
}
