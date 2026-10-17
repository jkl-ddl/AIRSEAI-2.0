# Personal VR deployment branch

The maintained branch is `personal/blindpeg-vr` in
<https://github.com/jkl-ddl/AIRSEAI-2.0>.
It starts at the original PR18 commit
`c214a247337dcf8c6d3eb1862d4e025320446d60`; the upstream history and `main`
remain available. No upstream pull request is part of this integration.

This branch contains the deployed OpenXR controller/head-camera fixes,
durable recording and recovery support, and the blind-peg VR physics fix.
The latter uses the existing TGS position-iteration setting (128 iterations)
only for VR collection of `blind_peg_insertion`. Gains, force limits, jaw
closure, simulation timestep, and expert/evaluation physics are preserved.
The effective physics settings are written to each collection report.

Native replay of recorded left/right controller inputs reduced stationary
grasp wrist drift from about 3 degrees to about 0.46 degrees. A controlled
50 mm forward move after a recorded grasp reached about 49.94 mm on both
arms while retaining the peg. Releasing the clutch still produced about
1.3 mm lateral drift. These are simulation replay results, not a claim of
zero drift or a new headset acceptance test.

Use `origin` for development and pushes. `upstream` is for explicitly
reviewed source updates only. Server and local checkouts should have the
same commit and no tracked-file differences after synchronization.
Runtime data, generated assets, Python environments and network/TLS
configuration stay outside the tracked source tree and are not part of
the equality claim. Never publish private access files in this fork.

For future updates: inspect local and server changes, commit and push to
the personal branch, deploy that exact commit, then verify tracked-file
hashes and relevant behavior. Preserve active collection and local data.
