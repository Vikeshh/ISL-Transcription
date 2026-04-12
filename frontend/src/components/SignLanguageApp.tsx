import { useRef, useEffect, useState, useCallback } from "react";
import { Progress } from "@/components/ui/progress";
import ConnectionStatus from "@/components/ConnectionStatus";
import PredictionHistory from "@/components/PredictionHistory";

interface Prediction {
  word: string;
  confidence: number;
  timestamp: number;
}

const SignLanguageApp = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);
  const waitingForAudio = useRef(false);

  const [connected, setConnected] = useState(false);
  const [currentWord, setCurrentWord] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [cameraReady, setCameraReady] = useState(false);

  const addToHistory = useCallback((word: string, conf: number) => {
    setHistory((prev) => [
      { word, confidence: conf, timestamp: Date.now() },
      ...prev.slice(0, 4),
    ]);
  }, []);

  // Start webcam
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setCameraReady(true);
        }
      })
      .catch((err) => console.error("Camera error:", err));

    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((t) => t.stop());
      }
    };
  }, []);

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
    const wsUrl = process.env.NODE_ENV === 'production'
    ? 'wss://isl-transcription-production.up.railway.app/ws'
    : 'ws://localhost:8000/ws'
    const ws = new WebSocket(wsUrl)      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (event) => {
        if (waitingForAudio.current) {
          // Audio bytes
          waitingForAudio.current = false;
          const blob = new Blob([event.data], { type: "audio/wav" });
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          audio.play().finally(() => URL.revokeObjectURL(url));
          return;
        }

        try {
          const data = JSON.parse(event.data);
          setCurrentWord(data.word || "—");
          setConfidence(Math.round((data.confidence || 0) * 100));
          if (data.word && data.word !== "—") {
            addToHistory(data.word, data.confidence);
          }
          if (data.has_audio) {
            waitingForAudio.current = true;
          }
        } catch {
          // ignore parse errors
        }
      };
    };

    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [addToHistory]);

  // Frame capture loop
  useEffect(() => {
    if (!cameraReady) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    intervalRef.current = window.setInterval(() => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) return;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(
        (blob) => {
          if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(blob);
          }
        },
        "image/jpeg",
        0.7
      );
    }, 100);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [cameraReady]);

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary">
            ISL → Speech
          </h1>
          <p className="text-sm text-muted-foreground">
            Indian Sign Language Recognition
          </p>
        </div>
        <ConnectionStatus connected={connected} />
      </header>

      <div className="mx-auto grid max-w-5xl gap-6 lg:grid-cols-[1fr_320px]">
        {/* Main area */}
        <div className="space-y-6">
          {/* Video */}
          <div className="relative overflow-hidden rounded-xl border border-border bg-card">
            <video
              ref={videoRef}
              className="w-full"
              autoPlay
              muted
              playsInline
              style={{ display: "block" }}
            />
            <canvas ref={canvasRef} className="hidden" />
            {!cameraReady && (
              <div className="flex aspect-video items-center justify-center text-muted-foreground">
                Waiting for camera…
              </div>
            )}
          </div>

          {/* Prediction */}
          <div className="rounded-xl border border-border bg-card p-6 text-center">
            <p className="mb-1 text-xs uppercase tracking-widest text-muted-foreground">
              Detected Sign
            </p>
            <p className="text-5xl font-black tracking-tight text-primary">
              {currentWord}
            </p>
            <div className="mx-auto mt-4 max-w-xs">
              <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                <span>Confidence</span>
                <span>{confidence}%</span>
              </div>
              <Progress value={confidence} className="h-2" />
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <PredictionHistory history={history} />
      </div>
    </div>
  );
};

export default SignLanguageApp;
