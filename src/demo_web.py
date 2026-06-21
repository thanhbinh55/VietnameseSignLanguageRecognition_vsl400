import base64
import csv
import json
import logging
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic, time
from urllib.parse import urlparse

import cv2
import numpy as np
from mediapipe.python.solutions import holistic
from simple_parsing import ArgumentParser
from transformers import Pipeline

from configs import InferenceConfig, ModelConfig
from data import Arm, get_sample_timestamp, ok_to_get_frame
from tools import Predictions, load_pipeline
from utils import POSE_BASED_MODELS, config_logger


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vietnamese Sign Language Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d9dee8;
      --text: #111827;
      --muted: #5b6472;
      --blue: #2563eb;
      --green: #15803d;
      --red: #b91c1c;
      --bar: #e8eef8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 20px 0;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 5px;
      color: var(--muted);
      font-size: 14px;
    }
    .status {
      min-width: 150px;
      padding: 9px 12px;
      border: 1px solid #b8caf6;
      background: #e8f0fe;
      color: #174ea6;
      border-radius: 8px;
      text-align: center;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      gap: 16px;
      align-items: stretch;
    }
    .video-panel, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .video-panel {
      padding: 10px;
      min-height: 520px;
    }
    .stage {
      position: relative;
      width: 100%;
      min-height: 500px;
      background: #111827;
      border-radius: 6px;
      overflow: hidden;
      display: grid;
      place-items: center;
    }
    video {
      width: 100%;
      height: 100%;
      max-height: 650px;
      object-fit: contain;
      background: #111827;
    }
    .overlay {
      position: absolute;
      left: 16px;
      right: 16px;
      bottom: 16px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
    }
    .pill {
      padding: 8px 10px;
      background: rgba(17, 24, 39, 0.76);
      color: white;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 650;
      max-width: 70%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    aside {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }
    .panel {
      padding: 14px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      border-radius: 8px;
      min-height: 40px;
      padding: 8px 12px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--blue);
      background: var(--blue);
      color: white;
    }
    button:disabled {
      opacity: 0.52;
      cursor: not-allowed;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 6px 0;
      color: var(--muted);
      border-bottom: 1px solid #eef1f6;
      font-size: 14px;
    }
    .metric strong {
      color: var(--text);
      font-weight: 700;
      text-align: right;
    }
    .prediction {
      margin: 10px 0 12px;
    }
    .prediction .label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 5px;
      font-weight: 700;
      font-size: 14px;
    }
    .bar {
      width: 100%;
      height: 8px;
      background: var(--bar);
      border-radius: 999px;
      overflow: hidden;
    }
    .fill {
      width: 0%;
      height: 100%;
      background: var(--blue);
      transition: width 140ms ease;
    }
    .history {
      flex: 1;
      min-height: 190px;
      overflow: auto;
    }
    .history-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .history-item {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid #eef1f6;
      font-size: 14px;
    }
    .history-item span:last-child {
      color: var(--green);
      font-weight: 700;
    }
    canvas { display: none; }
    @media (max-width: 900px) {
      header, main { display: block; }
      .status { margin-top: 12px; }
      aside { margin-top: 12px; }
      .video-panel { min-height: 360px; }
      .stage { min-height: 340px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Vietnamese Sign Language Recognition</h1>
        <div class="sub">SPOTER realtime demo | dua tay len de thu ky hieu, ha tay xuong de du doan</div>
      </div>
      <div id="status" class="status">Ready</div>
    </header>
    <main>
      <section class="video-panel">
        <div class="stage">
          <video id="video" autoplay muted playsinline></video>
          <div class="overlay">
            <div id="overlayText" class="pill">Camera chua bat dau</div>
            <div id="overlayBest" class="pill">--</div>
          </div>
        </div>
        <canvas id="canvas"></canvas>
      </section>
      <aside>
        <section class="panel">
          <h2>Dieu khien</h2>
          <div class="controls">
            <button id="startBtn" class="primary">Start camera</button>
            <button id="stopBtn" disabled>Stop</button>
            <button id="resetBtn">Reset</button>
            <button id="saveBtn">Save CSV</button>
          </div>
        </section>
        <section class="panel">
          <h2>Tin hieu</h2>
          <div class="metric"><span>Frames dang thu</span><strong id="frames">0</strong></div>
          <div class="metric"><span>Tay trai</span><strong id="leftAngle">--</strong></div>
          <div class="metric"><span>Tay phai</span><strong id="rightAngle">--</strong></div>
          <div class="metric"><span>Runtime</span><strong id="runtime">--</strong></div>
        </section>
        <section class="panel">
          <h2>Du doan</h2>
          <div id="predictions"></div>
        </section>
        <section class="panel history">
          <h2>Lich su</h2>
          <div id="history" class="history-list"></div>
        </section>
      </aside>
    </main>
  </div>
  <script>
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const statusEl = document.getElementById("status");
    const overlayText = document.getElementById("overlayText");
    const overlayBest = document.getElementById("overlayBest");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const resetBtn = document.getElementById("resetBtn");
    const saveBtn = document.getElementById("saveBtn");
    const predictionsEl = document.getElementById("predictions");
    const historyEl = document.getElementById("history");
    let stream = null;
    let running = false;
    let busy = false;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function renderPredictions(predictions) {
      predictionsEl.innerHTML = "";
      if (!predictions || predictions.length === 0) {
        predictionsEl.innerHTML = "<div class='metric'><span>Top 1</span><strong>--</strong></div>";
        overlayBest.textContent = "--";
        return;
      }
      predictions.forEach((pred, index) => {
        const score = Number(pred.score || 0) * 100;
        const row = document.createElement("div");
        row.className = "prediction";
        row.innerHTML = `
          <div class="label"><span>Top ${index + 1}: ${pred.gloss}</span><span>${score.toFixed(1)}%</span></div>
          <div class="bar"><div class="fill" style="width:${Math.max(0, Math.min(100, score))}%"></div></div>
        `;
        predictionsEl.appendChild(row);
      });
      overlayBest.textContent = `${predictions[0].gloss} ${(Number(predictions[0].score) * 100).toFixed(1)}%`;
    }

    function renderHistory(history) {
      historyEl.innerHTML = "";
      (history || []).slice(0, 12).forEach((item) => {
        const row = document.createElement("div");
        row.className = "history-item";
        row.innerHTML = `<span>${item.gloss}</span><span>${(Number(item.score) * 100).toFixed(1)}%</span>`;
        historyEl.appendChild(row);
      });
    }

    function renderState(data) {
      setStatus(data.status || "Running");
      overlayText.textContent = data.message || "--";
      document.getElementById("frames").textContent = data.frames || 0;
      document.getElementById("leftAngle").textContent = `${Number(data.left_angle || 0).toFixed(1)} deg`;
      document.getElementById("rightAngle").textContent = `${Number(data.right_angle || 0).toFixed(1)} deg`;
      document.getElementById("runtime").textContent = data.inference_time ? `${Number(data.inference_time).toFixed(2)}s` : "--";
      renderPredictions(data.predictions || []);
      renderHistory(data.history || []);
    }

    async function sendFrame() {
      if (!running || busy || video.readyState < 2) return;
      busy = true;
      const width = video.videoWidth;
      const height = video.videoHeight;
      const scale = Math.min(1, 720 / Math.max(width, height));
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", 0.74).split(",")[1];
      try {
        const response = await fetch("/api/frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image })
        });
        renderState(await response.json());
      } catch (error) {
        setStatus("Server error");
        overlayText.textContent = String(error);
      } finally {
        busy = false;
      }
    }

    function loop() {
      if (!running) return;
      sendFrame();
      setTimeout(loop, 120);
    }

    startBtn.addEventListener("click", async () => {
      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
        video.srcObject = stream;
      }
      await fetch("/api/reset", { method: "POST" });
      running = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus("Running");
      overlayText.textContent = "Dua tay len de bat dau";
      loop();
    });

    stopBtn.addEventListener("click", async () => {
      running = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      await fetch("/api/save", { method: "POST" });
      setStatus("Stopped");
    });

    resetBtn.addEventListener("click", async () => {
      const response = await fetch("/api/reset", { method: "POST" });
      renderState(await response.json());
    });

    saveBtn.addEventListener("click", async () => {
      const response = await fetch("/api/save", { method: "POST" });
      const data = await response.json();
      setStatus(data.status || "Saved");
    });

    renderPredictions([]);
  </script>
</body>
</html>
"""


class RealtimeRecognizer:
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig) -> None:
        self.model_config = model_config
        self.inference_config = inference_config
        self.inference_config.use_pose_model = model_config.arch in POSE_BASED_MODELS
        self.pipeline: Pipeline = load_pipeline(model_config, inference_config)
        self.detector = holistic.Holistic(model_complexity=0, min_detection_confidence=0.9)
        self.lock = threading.Lock()
        self.history = []
        self.reset(clear_history=True)

    def reset(self, clear_history: bool = True) -> dict:
        with self.lock:
            self.left_arm = Arm("left", self.inference_config.visibility)
            self.right_arm = Arm("right", self.inference_config.visibility)
            self.started_at = monotonic()
            self.frames = []
            self.last_predictions = []
            self.last_inference_time = 0.0
            self.last_clip = None
            self.last_clip_id = 0
            self.last_clip_pending = False
            self.message = "Dua tay len de bat dau"
            if clear_history:
                self.history = []
            return self._state("Ready")

    def process_frame(self, image_bytes: bytes) -> dict:
        with self.lock:
            array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None:
                return self._state("Frame error", "Khong doc duoc frame")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = self.detector.process(rgb_frame)
            if detection_results.pose_landmarks is None:
                self.message = "Khong thay nguoi trong khung hinh"
                return self._state("Running")

            self._update_capture(rgb_frame, detection_results)
            return self._state("Running")

    def _update_capture(self, rgb_frame: np.ndarray, detection_results) -> None:
        landmarks = detection_results.pose_landmarks.landmark
        self.left_arm.set_pose(landmarks)
        self.right_arm.set_pose(landmarks)

        elapsed_ms = int((monotonic() - self.started_at) * 1000)
        left_ok = ok_to_get_frame(
            arm=self.left_arm,
            angle_threshold=self.inference_config.angle_threshold,
            min_num_up_frames=self.inference_config.min_num_up_frames,
            min_num_down_frames=self.inference_config.min_num_down_frames,
            current_time=elapsed_ms,
            delay=self.inference_config.delay,
        )
        right_ok = ok_to_get_frame(
            arm=self.right_arm,
            angle_threshold=self.inference_config.angle_threshold,
            min_num_up_frames=self.inference_config.min_num_up_frames,
            min_num_down_frames=self.inference_config.min_num_down_frames,
            current_time=elapsed_ms,
            delay=self.inference_config.delay,
        )

        if left_ok or right_ok:
            self.frames.append(rgb_frame.copy())
            self.message = f"Dang thu ky hieu... {len(self.frames)} frames"

        start_time, end_time = get_sample_timestamp(self.left_arm, self.right_arm)
        if start_time != 0 and end_time != 0 and self.frames:
            self.message = "Dang du doan..."
            self._store_last_clip_preview(start_time, end_time)
            predictions = self._predict(start_time, end_time)
            self.last_predictions = predictions.predictions or []
            self.last_inference_time = predictions.inference_time
            self._append_history(predictions)
            self.frames = []
            self.left_arm.reset_state()
            self.right_arm.reset_state()
            self.message = "Dua tay len de thu ky hieu tiep theo"

    def _predict(self, start_time: float, end_time: float) -> Predictions:
        started_at = time()
        if self.inference_config.use_pose_model:
            sample = {
                "frames": self.frames,
                "fps": 25,
                "width": self.frames[0].shape[1],
                "height": self.frames[0].shape[0],
            }
        else:
            sample = np.array(self.frames)
        predictions = Predictions(predictions=self.pipeline(sample, top_k=self.inference_config.top_k))
        predictions.inference_time = time() - started_at
        predictions.start_time = start_time
        predictions.end_time = end_time
        return predictions

    def _store_last_clip_preview(self, start_time: float, end_time: float) -> None:
        if not self.frames:
            return

        max_preview_frames = 48
        if len(self.frames) <= max_preview_frames:
            indices = list(range(len(self.frames)))
        else:
            indices = np.linspace(0, len(self.frames) - 1, max_preview_frames, dtype=int).tolist()

        encoded_frames = []
        for idx in indices:
            rgb_frame = self.frames[idx]
            h, w = rgb_frame.shape[:2]
            scale = min(1.0, 360 / max(w, h))
            if scale < 1.0:
                preview = cv2.resize(
                    rgb_frame,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview = rgb_frame

            bgr_preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            ok, buffer = cv2.imencode(".jpg", bgr_preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ok:
                encoded_frames.append(base64.b64encode(buffer).decode("ascii"))

        self.last_clip_id += 1
        self.last_clip = {
            "id": self.last_clip_id,
            "frames": encoded_frames,
            "fps": 8,
            "frame_count": len(self.frames),
            "preview_frame_count": len(encoded_frames),
            "start_time": float(start_time),
            "end_time": float(end_time),
        }
        self.last_clip_pending = True

    def _append_history(self, predictions: Predictions) -> None:
        if not predictions.predictions:
            return
        best = predictions.predictions[0]
        self.history.insert(0, {
            "gloss": best["gloss"],
            "score": float(best["score"]),
            "inference_time": float(predictions.inference_time),
            "start_time": float(predictions.start_time),
            "end_time": float(predictions.end_time),
        })
        self.history = self.history[:40]

    def save_results(self) -> Path:
        output_dir = Path(self.inference_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "demo_web_results.csv"
        with output_file.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["start_time", "end_time", "inference_time", "gloss", "score"])
            for item in self.history:
                writer.writerow([
                    f"{item['start_time']:.3f}",
                    f"{item['end_time']:.3f}",
                    f"{item['inference_time']:.3f}",
                    item["gloss"],
                    f"{item['score']:.6f}",
                ])
        return output_file

    def _state(self, status: str, message: str | None = None) -> dict:
        state = {
            "status": status,
            "message": message or self.message,
            "frames": len(self.frames),
            "left_angle": float(self.left_arm.angle),
            "right_angle": float(self.right_arm.angle),
            "inference_time": float(self.last_inference_time),
            "predictions": [
                {"gloss": pred["gloss"], "score": float(pred["score"])}
                for pred in self.last_predictions
            ],
            "history": self.history,
        }
        if self.last_clip_pending and self.last_clip is not None:
            state["last_clip"] = self.last_clip
            self.last_clip_pending = False
        elif self.last_clip is not None:
            state["last_clip"] = {
                key: value
                for key, value in self.last_clip.items()
                if key != "frames"
            }
        return state


def make_handler(recognizer: RealtimeRecognizer):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path != "/":
                self.send_error(404)
                return
            self._send_html(HTML)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/frame":
                payload = self._read_json()
                try:
                    image = base64.b64decode(payload.get("image", ""))
                    response = recognizer.process_frame(image)
                except Exception as exc:
                    logging.exception("Frame processing failed")
                    response = {"status": "Error", "message": str(exc)}
                self._send_json(response)
                return
            if path == "/api/reset":
                self._send_json(recognizer.reset(clear_history=True))
                return
            if path == "/api/save":
                output_file = recognizer.save_results()
                self._send_json({"status": "Saved", "message": str(output_file)})
                return
            self.send_error(404)

        def log_message(self, format: str, *args) -> None:
            logging.debug(format, *args)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length))

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Run a local web demo for VSL recognition",
        add_config_path_arg=True,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_arguments(ModelConfig, "model")
    parser.add_arguments(InferenceConfig, "inference")
    return parser.parse_args()


def main(args: Namespace) -> None:
    config_logger(Path(args.inference.output_dir) / "demo_web.log")
    logging.info("Loading realtime recognizer")
    recognizer = RealtimeRecognizer(args.model, args.inference)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(recognizer))
    url = f"http://{args.host}:{args.port}"
    logging.info("Demo web app running at %s", url)
    print(f"Demo web app running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo web app")
    finally:
        recognizer.detector.close()
        server.server_close()


if __name__ == "__main__":
    main(get_args())
