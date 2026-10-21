# Ball number classification limitation

The available `billiards-model` v1 and v2 artifacts expose only generic `ball`
and `cue` classes. Exact numbers were being assigned after inference by an
OpenCV HSV color and stripe heuristic. That path wrote its guesses into the
tracked game state without gating on classifier confidence, so visually
plausible but unsupported numbers propagated to the UI and physics layer.

The live pipeline now leaves OpenCV identity refinement disabled by default.
It remains an explicit configuration opt-in for experiments, while generic
balls continue to be localized and tracked with `number: null`. The frontend
labels those detections as `Ball` rather than displaying a guessed identity.

Verified against the prerecorded `demo2_short.mp4` through the running video,
API, WebSocket, and Live View path: the API returned tracked detections with
`number: null`, and the Live View showed generic ball labels. No test suite was
run at the user's request. Reliable exact numbers require a separately trained
number classifier or a validated visual identity model.
