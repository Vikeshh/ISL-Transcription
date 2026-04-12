interface Prediction {
  word: string;
  confidence: number;
  timestamp: number;
}

const PredictionHistory = ({ history }: { history: Prediction[] }) => (
  <div className="rounded-xl border border-border bg-card p-5">
    <h2 className="mb-4 text-xs uppercase tracking-widest text-muted-foreground">
      Recent Predictions
    </h2>
    {history.length === 0 ? (
      <p className="text-sm text-muted-foreground">No predictions yet</p>
    ) : (
      <ul className="space-y-3">
        {history.map((p, i) => (
          <li
            key={p.timestamp}
            className="flex items-center justify-between rounded-lg bg-secondary px-3 py-2"
            style={{ opacity: 1 - i * 0.15 }}
          >
            <span className="font-semibold text-foreground">{p.word}</span>
            <span className="text-xs text-muted-foreground">
              {Math.round(p.confidence * 100)}%
            </span>
          </li>
        ))}
      </ul>
    )}
  </div>
);

export default PredictionHistory;
