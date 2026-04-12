const ConnectionStatus = ({ connected }: { connected: boolean }) => (
  <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs font-medium">
    <span
      className={`inline-block h-2 w-2 rounded-full ${
        connected ? "bg-primary animate-pulse" : "bg-destructive"
      }`}
    />
    <span className={connected ? "text-primary" : "text-destructive"}>
      {connected ? "Connected" : "Disconnected"}
    </span>
  </div>
);

export default ConnectionStatus;
