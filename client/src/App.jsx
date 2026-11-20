import { useEffect, useState } from 'react';

const SPLITS = ['train', 'valid', 'test1', 'test2', 'test3'];

export default function App() {
  const [split, setSplit] = useState('train');
  const [ids, setIds] = useState([]);
  const [encounterId, setEncounterId] = useState('');
  const [noteModels, setNoteModels] = useState([]);
  const [model, setModel] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [check, setCheck] = useState(null); // fact-check result
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState('');

  useEffect(() => {
    fetch('/api/note-models')
      .then((r) => r.json())
      .then((d) => {
        setNoteModels(d.models ?? []);
        setModel(d.default ?? d.models?.[0]?.id ?? '');
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    setIds([]);
    setEncounterId('');
    setResult(null);
    setCheck(null);
    fetch(`/api/encounters?split=${split}`)
      .then((r) => r.json())
      .then((d) => {
        setIds(d.encounter_ids ?? []);
        setEncounterId(d.encounter_ids?.[0] ?? '');
      })
      .catch((e) => setError(e.message));
  }, [split]);

  async function generate() {
    setLoading(true);
    setError('');
    setResult(null);
    setCheck(null);
    setCheckError('');
    try {
      const r = await fetch('/generate-note', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ split, encounter_id: encounterId, model }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'Request failed');
      setResult(data);
      setLoading(false);
      runCheck(data);
      return;
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }

  async function runCheck(data) {
    setChecking(true);
    try {
      const r = await fetch('/evaluate', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          generated_note: data.generated_note,
          reference_note: data.reference_note,
          dialogue: data.dialogue,
        }),
      });
      const out = await r.json();
      if (!r.ok) throw new Error(out.detail || 'Fact check failed');
      setCheck(out);
    } catch (e) {
      setCheckError(e.message);
    }
    setChecking(false);
  }

  return (
    <main>
      <h1>Clinical Note Comparison</h1>

      <div className="controls">
        <select value={split} onChange={(e) => setSplit(e.target.value)} disabled={loading}>
          {SPLITS.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select value={encounterId} onChange={(e) => setEncounterId(e.target.value)} disabled={loading}>
          {ids.map((id) => (
            <option key={id}>{id}</option>
          ))}
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)} disabled={loading} title="Model">
          {noteModels.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <button onClick={generate} disabled={loading || !encounterId || !model}>
          {loading ? 'Generating…' : 'Generate'}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {(result || loading) && (
        <div className="columns">
          <section>
            <h2>LLM Generated Note{result && ` (${noteModels.find((m) => m.id === result.model)?.label ?? result.model})`}</h2>
            <pre>{result ? result.generated_note : 'Generating…'}</pre>
          </section>
          <section>
            <h2>Dataset Note (Reference)</h2>
            <pre>{result ? result.reference_note : 'Loading…'}</pre>
          </section>
        </div>
      )}

      {result && (
        <section className="factcheck">
          <h2>Fact Check</h2>
          <p className="hint">
            Key facts from the dataset note (treated as ground truth), checked against the LLM note.
          </p>
          {checking && <p>Checking facts… this can take a little while.</p>}
          {checkError && <p className="error">{checkError}</p>}
          {check && <FactCheck check={check} />}
          {check?.precision && <Precision precision={check.precision} />}
        </section>
      )}
    </main>
  );
}

const LABELS = { correct: 'Correct', missed: 'Missed', wrong: 'Wrong', unjudged: 'Unjudged' };

function FactCheck({ check }) {
  const { counts, total, sections } = check;
  return (
    <>
      <div className="summary">
        {['correct', 'missed', 'wrong'].map((v) => (
          <span key={v} className={`badge ${v}`}>
            {LABELS[v]}: {counts[v]}
          </span>
        ))}
        <span className="total">
          {counts.correct} of {total} facts correct
        </span>
      </div>
      {Object.entries(sections).map(([name, rows]) => (
        <div key={name} className="fc-section">
          <h3>{name}</h3>
          <table>
            <thead>
              <tr>
                <th>Fact (from dataset note)</th>
                <th>LLM output</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={r.verdict}>
                  <td>{r.fact}</td>
                  <td>
                    <span className={`badge ${r.verdict}`}>{LABELS[r.verdict]}</span>
                  </td>
                  <td>{r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

const P_LABELS = {
  in_reference: 'In reference',
  in_dialogue: 'In dialogue only',
  hallucinated: 'Hallucinated',
  unjudged: 'Unjudged',
};

function Precision({ precision }) {
  const { counts, total, sections } = precision;
  const supported = counts.in_reference + counts.in_dialogue;
  return (
    <>
      <h2 className="p-title">Hallucination Check</h2>
      <p className="hint">
        Facts from the LLM note, checked against the dataset note first, then against the dialogue.
      </p>
      <div className="summary">
        {['in_reference', 'in_dialogue', 'hallucinated'].map((v) => (
          <span key={v} className={`badge ${v}`}>
            {P_LABELS[v]}: {counts[v]}
          </span>
        ))}
        <span className="total">
          {supported} of {total} LLM facts supported
        </span>
      </div>
      {Object.entries(sections).map(([name, rows]) => (
        <div key={name} className="fc-section">
          <h3>{name}</h3>
          <table>
            <thead>
              <tr>
                <th>Fact (from LLM note)</th>
                <th>Result</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={r.verdict}>
                  <td>{r.fact}</td>
                  <td>
                    <span className={`badge ${r.verdict}`}>{P_LABELS[r.verdict]}</span>
                  </td>
                  <td>{r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}
