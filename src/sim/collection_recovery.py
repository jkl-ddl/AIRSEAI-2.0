"""CPU-only recovery of a stopped collection batch, without altering its sources."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import uuid


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _vendor_api():
    # Deliberately lazy: this module never imports Isaac, torch, Kit or GPU APIs.
    from lerobot_tactile_ext.merge import inspect_recording, merge_recordings
    from lerobot_tactile_ext.recovery import _candidate_episodes, recover_recording

    return _candidate_episodes, recover_recording, inspect_recording, merge_recordings


def _counts(info):
    episodes, frames = info.get("total_episodes"), info.get("total_frames")
    if type(episodes) is not int or type(frames) is not int or episodes <= 0 or frames <= 0:
        raise ValueError("Recording has invalid episode/frame counts")
    return episodes, frames


def _discarded(row):
    return row.get("status") in ("discarded", "reset") or row.get("discarded") is True


def recover_session(session_dir):
    """Recover one stopped batch. Partial clips remain separate and require review.

    A completed recovery.json is the idempotence boundary. Original attempt files
    and SESSION/dataset are read only. Failed retries use a fresh output directory.
    The caller must stop the original collector before invoking this function.
    """
    session = Path(session_dir).expanduser().resolve()
    if not session.is_dir():
        raise ValueError("Session directory does not exist")
    summary_path = session / "recovery.json"
    if summary_path.exists():
        previous = _read(summary_path)
        if previous.get("status") == "complete":
            return previous
    summary = dict(status="running", session_dir=str(session), recovered_count=0, recovered_frames=0,
                   retained_count=0, message="正在核验并恢复已落盘数据，原始批次保持不变。",
                   paths=dict(interrupted=[], completed=None, retained_sources=[]), errors=[],
                   training_eligible=False)
    _atomic_json(summary_path, summary)
    output = None

    def output_root():
        nonlocal output
        if output is None:
            name = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
            parent = session / "recovered"
            parent.mkdir(exist_ok=True)
            output = parent / name
            output.mkdir(exist_ok=False)
            summary["output_dir"] = str(output)
        return output

    def error(path, exc):
        summary["errors"].append(dict(source=str(path.relative_to(session)),
                                      error_type=type(exc).__name__, message=str(exc)[:500]))

    try:
        candidates, recover, inspect, merge = _vendor_api()
        completed = []
        provenance = []
        retained = 0
        attempts = session / "report" / "attempts"
        for folder in sorted(attempts.iterdir() if attempts.is_dir() else [], key=lambda p: p.name):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            try:
                result = _read(folder / "result.json") if (folder / "result.json").exists() else {}
                intent = _read(folder / "write_intent.json") if (folder / "write_intent.json").exists() else {}
                row = intent.get("row", {})
                if not isinstance(row, dict):
                    raise ValueError("Invalid write-intent row")
                # Explicit human Y/reset decisions dominate every other piece of evidence.
                if _discarded(result) or _discarded(row):
                    continue
                raw = folder / "raw"
                info_path = raw / "meta" / "info.json"
                saved = result.get("saved") is True and result.get("write_status") == "complete"
                intended = intent.get("keep") is True
                if info_path.exists() and (saved or intended):
                    checked = inspect(raw)
                    episodes, frames = _counts(checked["info"])
                    if episodes != 1:
                        raise ValueError("Expected one completed episode per attempt")
                    proof = result if saved else row
                    if intended and not saved and (type(proof.get("frames")) is not int or proof["frames"] != frames):
                        raise ValueError("Completed raw frame count differs from save intent")
                    if saved and "frames" in proof and proof["frames"] != frames:
                        raise ValueError("Completed raw frame count differs from result")
                    completed.append(raw)
                    provenance.append(dict(attempt_id=int(folder.name), source=str(raw), proof=proof))
                    retained += episodes
                    summary["paths"]["retained_sources"].append(str(raw))
                    # A finalized spool can remain after a crash before its cleanup.
                    continue
                spool = raw / ".recovery"
                if not spool.exists():
                    if saved or intended or info_path.exists():
                        raise ValueError("No verifiable completed recording or unfinished frame spool")
                    continue
                if not (spool / "session.json").exists() and not any(spool.glob("episode-*/frames/*/commit.json")):
                    # Death during spool creation, before the first committed frame.
                    continue
                spool_session = _read(spool / "session.json")
                if spool_session.get("version") != 1:
                    raise ValueError("Unsupported recovery spool version")
                if spool_session.get("status") == "finalized":
                    raise ValueError("Finalized spool has no verified completed recording")
                episodes_to_recover = list(candidates(spool, spool_session))
                readable = {item[0]: item[2] for item in episodes_to_recover}
                for episode in spool.glob("episode-*"):
                    try:
                        state = _read(episode / "episode.json")
                        if state.get("decision") == "discard" or state.get("status") == "discarded":
                            continue
                        declared = state.get("count", 0)
                        if type(declared) is not int or declared < 0:
                            raise ValueError("Invalid source episode count")
                        if declared > readable.get(episode, 0):
                            raise ValueError("Some published source frames are unreadable; committed prefix may still be recovered")
                    except Exception as exc:
                        error(episode, exc)
                expected_episodes = len(episodes_to_recover)
                expected_frames = sum(item[2] for item in episodes_to_recover)
                if not expected_episodes or not expected_frames:
                    continue
                destination = output_root() / "interrupted" / folder.name
                recover(raw, destination)
                actual = _read(destination / "meta" / "info.json")
                recovered_episodes, recovered_frames = _counts(actual)
                if (recovered_episodes, recovered_frames) != (expected_episodes, expected_frames):
                    raise ValueError("Recovered dataset counts differ from committed source frames")
                recovery_meta = actual.get("recovery", {})
                if (recovery_meta.get("interrupted") is not True or recovery_meta.get("needs_review") is not True
                        or recovery_meta.get("training_eligible") is not False or actual.get("splits") != {}):
                    raise ValueError("Recovered clips must be review-only with empty training splits")
                summary["recovered_count"] += recovered_episodes
                summary["recovered_frames"] += recovered_frames
                summary["paths"]["interrupted"].append(str(destination))
                _atomic_json(summary_path, summary)
            except Exception as exc:
                error(folder, exc)
        summary["retained_count"] = retained
        dataset = session / "dataset"
        existing = False
        completed_frames = sum(_counts(_read(raw / "meta" / "info.json"))[1] for raw in completed)
        if (dataset / "meta" / "info.json").exists():
            try:
                existing_info = inspect(dataset)["info"]
                existing_count, existing_frames = _counts(existing_info)
                summary["retained_count"] = max(retained, existing_count)
                summary["paths"]["completed"] = str(dataset)
                # A valid older/subset merge is not evidence that every verified
                # completed attempt is included. Preserve it, but publish a new
                # complete set rather than silently omitting later attempts.
                existing = not completed or (existing_count, existing_frames) == (retained, completed_frames)
                if not existing:
                    summary["paths"]["original_completed"] = str(dataset)
                    error(dataset, ValueError("Existing completed batch counts differ from verified completed attempts; "
                                              "original batch retained and verified attempts rebuilt separately"))
            except Exception as exc:
                error(dataset, exc)
        if completed and not existing:
            try:
                destination = output_root() / "completed"
                if destination.exists():
                    raise FileExistsError("Completed recovery output already exists")
                merge(completed, destination)
                merged_count, merged_frames = _counts(inspect(destination)["info"])
                if (merged_count, merged_frames) != (retained, completed_frames):
                    raise ValueError("Merged completed recording counts differ from verified sources")
                summary["paths"]["completed"] = str(destination)
                _atomic_json(output_root() / "completed_provenance.json", dict(sources=provenance,
                             note="Original criterion labels are preserved; interrupted recovery clips are separate."))
            except Exception as exc:
                error(session / "recovered", exc)
        summary["status"] = "error" if summary["errors"] else "complete"
        summary["message"] = ("部分源数据未通过核验，原始数据和已恢复输出均保留，请联系管理员检查。" if summary["errors"] else
                              "已恢复可用片段并保留完成记录；未完成片段须人工审核，原始数据保持不变。")
    except Exception as exc:
        error(session, exc)
        summary.update(status="error", message="恢复失败，原始数据保持不变，请联系管理员检查。")
    _atomic_json(summary_path, summary)
    return summary
