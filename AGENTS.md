# Personal development and deployment

- Work on `personal/blindpeg-vr` in `jkl-ddl/AIRSEAI-2.0`. `origin` is the personal fork; `upstream` is read-only. Do not open upstream pull requests or push upstream without an explicit user request.
- Preserve the validated PR18-based runtime. Do not replace it with upstream `main` as part of routine synchronization.
- Keep credentials, certificates, environments, datasets, generated robot assets, logs, and model weights outside Git.
- Before synchronizing a running machine, compare its tracked-file hashes with the intended commit. Preserve any divergent or untracked work; do not reset or clean it away.
- Deploy from a concrete personal-fork commit, verify the tracked source matches, and record the commit. Do not restart an active collection or unrelated GPU task merely to update Git metadata or documentation.
- The legacy `work/vr-debug-20261016/pr18-alignment` copy is a historical staging artifact. Use this checkout for future project changes.
- Network/NUC/CloudXR service configuration is managed separately; a source sync does not authorize changing those services.
