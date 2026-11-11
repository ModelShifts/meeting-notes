Backend: api · transcription: faster-whisper large-v3-turbo · notes: qwen2.5-7b-instruct · speaker labels: sherpa-onnx (pyannote-segmentation-3.0 + CAM++)

Speech context: prompt='' hotwords='sprint, planning, backlog, ticket, bug, fix, feature, prototype, code review, merge, deploy, release, build, test plan, regression, login, password, search, performance, cache, database, backend, frontend, API, meeting, onboarding, design, final, deadline, roadmap, board, demo, feedback'

| Sample | Lang | Transcript error | Speaker time correct | Speakers found | Action items found | Owner right | Due right | Precision | Decisions (listed/script) | Time (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| design-review-zh.wav | zh | CER 2.0% | 99% | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 | 1/2 | 93 | - |
| sprint-planning-mixed.wav | mixed | MER 1.1% | 97% | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 | 1/1 | 101 | - |
| weekly-sync-en.wav | en | WER 0.6% | 96% | 2/2 | 4/4 | 4/4 | 3/4 | 4/4 | 1/1 | 191 | due None for 'Ask the design team for the new pictures for the app store' (want today) |
| **All** | | | | | **10/10 (100% recall)** | 10/10 | 9/10 | 10/10 | | | |
