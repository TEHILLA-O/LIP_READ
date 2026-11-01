# Evaluation

How accuracy is measured here, what the current numbers mean, and how to get
them yourself.

## Two questions, two modes

`scripts/evaluate_lrs3.py` has two modes because a bad transcript can come from
two very different places.

- **`--source crops`** feeds LRS3's own preprocessed 96x96 mouth crops straight
  to the model, skipping this project's detection and alignment. It answers:
  are the weights, tensor layout, vocabulary and beam search wired up
  correctly? Near 0% WER would mean the labels are leaking; above about 50%
  means the model path is broken.
- **`--source pipeline`** runs whole video files through the real chain:
  decode, detect, track, align, crop, infer, decode. It answers: does our
  preprocessing keep what the model needs? The gap between the two modes is
  the cost of our preprocessing. If it grows, look at the debug MP4s.

The pipeline mode needs videos with a known transcript but no real recordings
of anyone. `scripts/make_talking_face_video.py` builds them by grafting an LRS3
mouth sequence back into a full scene through the inverse of the alignment
transform. These composite clips read harder than real video, because one
person's mouth sits on another face and the frames are resampled twice.

## Current results

| Mode | Utterances | WER |
| --- | --- | --- |
| `--source crops` | 5 | 5.7% (2 of 35 words wrong) |
| `--source pipeline` | 5 | 25.7% (9 of 35 words wrong) |
| Published Auto-AVSR | full LRS3 test set | 20.3% (upstream figure, not run here) |

Both rows were rerun on 1 November 2026 on a CPU-only Linux machine with
Python 3.11 and gave the same numbers.

Read these with care. Five utterances is a check that the system works end to
end, not a benchmark. The crops figure being lower than the published 20.3% is
the luck of five short, clear sentences, not a sign this setup beats the paper.

Not measured yet:

- WER on a larger LRS3 sample
- Accuracy on real webcam footage
- Live mode latency and accuracy end to end
- Whether the confidence score is calibrated

## Reproduce

With the environment, reference checkout and checkpoint from the README's
quick start in place:

```bash
python scripts/fetch_lrs3_samples.py --count 5      # downloads 5 LRS3 test samples
python scripts/evaluate_lrs3.py --source crops

python scripts/make_talking_face_video.py           # builds composite clips from those samples
python scripts/evaluate_lrs3.py --source pipeline
```

`fetch_lrs3_samples.py` downloads 4 samples unless you pass `--count`, so use
`--count 5` to match the table. It only pulls the parts of the parquet shard it
needs from the `mattymchen/lrs3-test` dataset on Hugging Face. Samples land in
`data/test_videos/lrs3/` and composites in `data/test_videos/composite/`; both
are git-ignored because they show real people's faces.

Add `--device cpu` or `--device cuda` to force a device, and `--beam-size N` to
try a different beam width.

## Other checks

| Check | Command |
| --- | --- |
| Geometry, frame rates, tensor shapes, API, streaming | `python -m pytest` |
| Watch exactly what the model sees | `python scripts/transcribe.py CLIP --debug` |
| A grid of frames from a clip | `python scripts/contact_sheet.py CLIP` |
| Retune the live speech thresholds | `python scripts/tune_motion_thresholds.py` |

`tune_motion_thresholds.py` builds its still, head-moving and talking clips in
memory and also uses the composite clips, so run it from the repo root after
`make_talking_face_video.py`. When it was rerun on
1 November 2026 against freshly generated fixtures, the best setting it found
caught 4 of the 5 composite speech clips with no false triggers (start 0.012,
stop 0.008), slightly different from the shipped defaults (0.016, 0.008). The
defaults have not been changed; treat them as due for a retune on a bigger set
of clips.

The debug export is the fastest way to catch a preprocessing regression: a crop
that has drifted off the mouth is obvious in two seconds of video and nearly
invisible in a WER number.
