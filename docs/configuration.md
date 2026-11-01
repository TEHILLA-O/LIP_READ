# Configuration

Everything is set through environment variables, so nothing needs editing in
code. Set them in the shell that starts the backend. The service also reads a
`.env` file from the folder it starts in (`services/ml-api`) for the
`VSR_API_*` settings.

## Model and decoding

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_MODEL` | `auto_avsr` | Which registered adapter to load |
| `VSR_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `VSR_BEAM_SIZE` | `20` | Beam search width. Lower is faster and slightly less accurate |
| `VSR_CTC_WEIGHT` | `0.1` | Weight of the CTC score against the attention decoder |
| `VSR_N_BEST` | `3` | Hypotheses kept: the best one plus alternatives |

## Preprocessing

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_TARGET_FPS` | `25` | Leave at 25. The model has no other time base (see the [preprocessing spec](preprocessing-spec.md)) |
| `VSR_DETECTION_LONG_SIDE` | `640` | Frames are scaled to this long side for face detection, to cap cost |
| `VSR_MAX_FRAMES` | `1500` | Longest clip processed, in frames (60 s at 25 fps) |
| `VSR_MIN_DETECTION_RATIO` | `0.3` | Below this share of frames with a face, a clip is reported as unreadable |

## Live streaming

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_BUFFER_SECONDS` | `8.0` | Rolling frame buffer length |
| `VSR_MIN_UTTERANCE_SECONDS` | `1.2` | Shortest live segment worth transcribing |
| `VSR_MAX_UTTERANCE_SECONDS` | `4.0` | Longest live segment before it is cut |
| `VSR_MOTION_START_THRESHOLD` | `0.016` | Lip-aperture change that opens an utterance |
| `VSR_MOTION_STOP_THRESHOLD` | `0.008` | Change below which the mouth counts as still |
| `VSR_MAX_QUEUED_JOBS` | `2` | Inference queue depth before the oldest segment is dropped |

The two motion thresholds came from `scripts/tune_motion_thresholds.py`. If
you change them, rerun that script against the fixtures rather than guessing.

## Privacy and debug output

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_PERSIST_UPLOADS` | `0` | Keep uploaded media after inference |
| `VSR_ALLOW_DEBUG_EXPORT` | `1` | Allow debug MP4 exports |

## API service

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_API_CORS_ORIGINS` | `["http://localhost:3000", "http://127.0.0.1:3000"]` | JSON list of web origins allowed to call the API |
| `VSR_API_MAX_UPLOAD_BYTES` | `209715200` (200 MB) | Upload limit, enforced while the file is written |
| `VSR_API_MAX_WEBSOCKET_MESSAGE_BYTES` | `4194304` (4 MB) | Largest single live frame accepted |
| `VSR_API_STATUS_INTERVAL_FRAMES` | `5` | How often the live endpoint sends a `status` message |
| `VSR_API_LOG_LEVEL` | `INFO` | Log level |

## Paths

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_REFERENCES_DIR` | `references/` | Where the `auto_avsr` checkout lives |
| `VSR_CHECKPOINTS_DIR` | `checkpoints/` | Where model weights are stored |
| `VSR_DEBUG_DIR` | `debug_out/` | Where `scripts/transcribe.py --debug` writes MP4s |
| `VSR_TEST_VIDEO_DIR` | `data/test_videos/` | Generated fixtures and LRS3 samples |

## Web app

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEXT_PUBLIC_ML_API_URL` | `http://127.0.0.1:8000` | Backend URL. Put it in `apps/web/.env.local` |
