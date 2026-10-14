// What the engine thought about the position: values plus prior / visits / Q per column.

const signed = (x) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${x.toFixed(3)}`);

export default function SearchPanel({ analysis }) {
  const searched = analysis && analysis.simulations > 0;
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
            <div>
              <div className="value">{signed(analysis.net_value)}</div>
              <div className="meta">network value</div>
            </div>
          </div>
          <div className="meta">
            for the side to move, −1 lost … +1 won ·{' '}
            {searched ? `${analysis.simulations} simulations` : 'no search (policy only)'}
            {analysis.sampled && ' · move was sampled'}
          </div>

          <table>
            <thead>
              <tr>
                <th>col</th>
                <th>P</th>
                <th>N</th>
                <th>Q</th>
              </tr>
            </thead>
            <tbody>
              {analysis.moves.map((m) => (
                <tr key={m.move}>
                  <td>{m.move}</td>
                  <td>{m.prior.toFixed(3)}</td>
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
      <p className="note">
        <b>P</b> is the network's prior before searching, <b>N</b> is where the search spent its visits,{' '}
        <b>Q</b> is the value it found for that move. When N disagrees with P, the search is correcting the
        network. Keep the simulations fixed when comparing checkpoints.
      </p>
    </section>
  );
}
