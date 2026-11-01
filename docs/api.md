# API

The backend in `services/ml-api` is a FastAPI app. Run it from that folder:

```bash
../../.venv/bin/python -m uvicorn app.main:app --port 8000
```

FastAPI's own interactive docs are then at http://127.0.0.1:8000/docs.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Whether the model is loaded, on which device, and why not if it failed |
| `GET /v1/models` | Registered adapters, which are implemented, and the active model's input contract |
| `POST /v1/transcribe` | Transcribe an uploaded video. `?debug=true` also exports MP4s |
| `POST /v1/preprocess` | Preprocessing only, to check framing without the model. Debug export is on by default |
| `GET /v1/debug/{token}/{kind}` | Download a debug MP4 (`mouth_roi` or `network_input`) until it expires |
| `WS /v1/stream` | Live frames in, transcripts out |

Uploads go in a multipart field called `file`:

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@clip.mp4;type=video/mp4" "http://127.0.0.1:8000/v1/transcribe?debug=true"
```

Accepted types are MP4, QuickTime, WebM, Matroska, AVI and
`application/octet-stream`, up to 200 MB by default.

| Status | When |
| --- | --- |
| 400 | Empty file, or a malformed debug token |
| 403 | Debug export requested but disabled (`VSR_ALLOW_DEBUG_EXPORT=0`) |
| 404 | Unknown debug artefact, or the debug clip has expired |
| 413 | Upload larger than `VSR_API_MAX_UPLOAD_BYTES` |
| 415 | Unsupported content type |
| 422 | The video could not be read |

A video that reads fine but has no usable face is **not** an HTTP error. It
comes back as a normal result with empty `text` and an `error` message.

## The result object

Every transcription, from upload or live, has the same shape:

```json
{
  "text": "I WOULD LOVE TO HEAR FROM YOU",
  "confidence": 0.792,
  "alternatives": [
    { "text": "I WOULD LOVE TO GET IT FROM YOU", "confidence": 0.6527 },
    { "text": "I WOULD LOVE TO HEAR IT FROM YOU", "confidence": 0.5583 }
  ],
  "processing_ms": 779,
  "frames_processed": 34,
  "mouth_detected": true,
  "error": null,
  "diagnostics": { "inference_ms": 575, "detection_ratio": 1.0, "source_fps": 25.0 },
  "debug": null
}
```

| Field | Meaning |
| --- | --- |
| `text` | Best hypothesis, upper case. Empty when there is no transcript |
| `confidence` | 0 to 1. A ranking score, not a calibrated probability |
| `alternatives` | Next-best hypotheses (up to `VSR_N_BEST` minus one) |
| `processing_ms` | Total time for the request on the server |
| `frames_processed` | Frames after resampling to 25 fps |
| `mouth_detected` | Whether a mouth was found in enough frames |
| `error` | Why there is no transcript: no face, clip too short, model not loaded |
| `diagnostics` | Timings and preprocessing detail for the debug view (trimmed above) |
| `debug` | Links to debug MP4s when requested |

The service never substitutes plausible text for a failure.

## Live WebSocket: `/v1/stream`

The client sends:

- **Binary messages**: one JPEG-encoded frame each, at 25 fps. Frames over
  4 MB are rejected.
- **Text messages**: JSON control commands.

| Command | Effect |
| --- | --- |
| `{"type": "ping"}` | Replies `{"type": "pong"}` |
| `{"type": "flush"}` | Transcribe whatever utterance is open now, for when the speaker stops or the tab closes |
| `{"type": "reset"}` | Clear the buffer, segmenter and tracker |
| `{"type": "close"}` | Close the connection |

The server sends JSON messages with a `type` field:

| Type | When | Main fields |
| --- | --- | --- |
| `ready` | On connect | `model`, `model_loaded`, `error`, `spec`, `config` (fps, utterance limits, buffer length) |
| `status` | Every 5 frames | Face and mouth boxes, lip outline, motion energy, buffer and queue stats |
| `segment` | An utterance was cut and queued | `sequence`, `reason`, `duration`, `frames`, `accepted`, `displaced_earlier_segment` |
| `transcript` | Inference finished | `sequence`, `queue_wait_ms`, `dropped`, plus the result object above |
| `reset` | After a reset command | |
| `error` | Bad frame, bad command or internal failure | `message` |

`apps/web/src/lib/useLiveSession.ts` is a working client to copy from.
