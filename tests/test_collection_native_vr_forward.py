"""Exercise the native Collector device call without starting Isaac Sim."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "sim" / "collection.py"


def _load_blindpeg_task(monkeypatch):
    """Use the real task configuration, replacing only simulator dependencies."""
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def scene(**_):
        return types.SimpleNamespace(
            robot=types.SimpleNamespace(spawn=types.SimpleNamespace(articulation_props=types.SimpleNamespace(
                solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ))),
            board=types.SimpleNamespace(init_state=types.SimpleNamespace(pos=(0, 0, 0))),
            peg=types.SimpleNamespace(init_state=types.SimpleNamespace(pos=(0, 0, 0))),
            set_clearance=lambda _: 0.012,
            set_light_scale=lambda _: 1.0,
        )

    stub("torch")
    stub("sim.envs.bimanual_tactile", TABLE_TOP_Z=0.0, gel_gap=lambda x: x)
    stub("sim.envs.blind_peg_insertion", BOARD_TOP_Z=0.04, DEFAULT_LIGHT_SCALE=1.0,
         HOLE_JITTER=0.0, HOLE_XY=(0, 0), PEG_SPAWN_XY=(0, 0), SPAWN_JITTER=0.0,
         BlindPegInsertionSceneCfg=scene)
    stub("sim.tactile", apply_ccd=None, apply_sdf_collision=None, apply_triangle_mesh_collision=None)
    stub("sim.tactile.quantities", quat_apply=None)
    spec = importlib.util.spec_from_file_location(
        "blindpeg_solver_under_test", SOURCE.parent / "tasks" / "blind_peg_insertion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASK


def _load_collection(monkeypatch, calls):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        if name in {"sim", "sim.envs", "sim.manager_based", "sim.manager_based.common", "lerobot_tactile_ext"}:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("lerobot_tactile_ext")
    stub("lerobot_tactile_ext.merge", merge_recordings=lambda *_: None)
    sim = stub("sim")
    sim.init_state = stub("sim.init_state")
    stub("sim.collection_io", EpisodeWrite=object, EpisodeWriteQueue=object, write_json=lambda *_: None)
    stub("sim.envs")
    stub("sim.envs.bimanual_tactile", TACTILE_SENSORS=())
    stub("sim.harness", CTRL_FPS=20)
    stub("sim.manager_based", make_env=lambda *_a, **_kw: None)
    stub("sim.manager_based.common")
    stub("sim.manager_based.common.mdp", hold_action=lambda *_: None)
    stub("sim.manager_based.common.slot", SlotHarness=object)
    stub(
        "sim.manager_based.common.states",
        prepare_restore=lambda *_: None,
        restore_states=lambda *_: None,
        snapshot_slot=lambda *_: None,
    )
    stub("sim.recording", EpisodeRecorder=object)
    stub("sim.recording_health", RecordingHeartbeat=lambda *_a, **_kw: types.SimpleNamespace(
        emit=lambda *_a, **_kw: None, run_io=lambda _phase, function: function()
    ))

    class Device:
        def begin_episode(self, ctx):
            self.ctx = ctx

        def help_text(self):
            return "fake input device"

    def make_device(name, task, harness, **kwargs):
        calls.append((name, task, harness, kwargs))
        return Device()

    stub("sim.teleop", make_device=make_device)
    stub("sim.video", Mp4Writer=object, compose_grid_frame=lambda *_a, **_kw: None)
    stub(
        "sim.xr_idle",
        render_completed_scene=lambda *_a, **_kw: None,
        should_render_completed_scene=lambda *_a, **_kw: False,
    )
    spec = importlib.util.spec_from_file_location("collection_native_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "state_hash", lambda _: "sha")
    return module


def test_native_collector_forwards_position_gain_only_for_vr(monkeypatch):
    calls = []
    module = _load_collection(monkeypatch, calls)

    def begin(device_name, gain):
        args = types.SimpleNamespace(
            seed=0, teleop_device=device_name, pos_speed=0.2, rot_speed=0.9,
            jaw_speed=0.02, vr_joint_speed_deg_s=60.0,
        )
        if gain is not None:
            args.vr_position_gain = gain
        task, harness = object(), object()
        collector = types.SimpleNamespace(
            args=args, handles=[harness], task=task, knobs={}, snapshots=[],
            dataset=None, teleop=True, controller=None,
        )
        actual = {"id": {"episode": 0}}

        def snapshot_start(attempt):
            attempt.ctx = {"side": "right"}
            return actual

        collector.snapshot_start = snapshot_start
        attempt = types.SimpleNamespace(
            requested={"meta": {}}, attempt_id=0, env_id=0,
            report={}, folder=Path("."), ctx=None,
        )
        module.Collector.begin(collector, attempt)
        assert attempt.mode == "running"
        assert attempt.controller.ctx == {"side": "right"}
        return task, harness

    vr_task, vr_harness = begin("vr", 2.0)
    assert calls[-1][:3] == ("vr", vr_task, vr_harness)
    assert calls[-1][3]["vr_position_gain"] == 2.0
    assert calls[-1][3]["vr_joint_speed_deg_s"] == 60.0

    begin("vr", None)
    assert calls[-1][3]["vr_position_gain"] == 1.0

    begin("keyboard", 2.0)
    assert "vr_position_gain" not in calls[-1][3]
    assert "vr_joint_speed_deg_s" not in calls[-1][3]


@pytest.mark.parametrize(
    "task_name,device,teleop,explicit,expected",
    [
        ("blind_peg_insertion", "vr", True, None, 128),
        ("blind_peg_insertion", "keyboard", True, None, 12),
        ("blind_peg_insertion", "scripted", True, None, 12),
        ("blind_peg_insertion", "scripted", False, None, 12),
        ("stack_cube", "vr", True, None, 12),
        *[("blind_peg_insertion", device, teleop, count, count)
          for count in (12, 128)
          for device, teleop in (("vr", True), ("scripted", True), ("scripted", False))],
    ],
)
def test_grasp_solver_profile_reaches_native_environment(
    monkeypatch, tmp_path, task_name, device, teleop, explicit, expected
):
    if task_name == "blind_peg_insertion":
        task = _load_blindpeg_task(monkeypatch)
        knobs = {name: knob.default for name, knob in task.knobs.items()}
        knobs["robot_position_iterations"] = explicit
    module = _load_collection(monkeypatch, [])
    props = types.SimpleNamespace(
        solver_position_iteration_count=12, solver_velocity_iteration_count=1
    )
    cfg = types.SimpleNamespace(
        robot=types.SimpleNamespace(spawn=types.SimpleNamespace(articulation_props=props))
    )
    if task_name != "blind_peg_insertion":
        task = types.SimpleNamespace(name=task_name, horizon_s=180, make_scene_cfg=lambda _: cfg)
        knobs = {}
    args = types.SimpleNamespace(
        num_envs=1, seed=0, episodes=1, dataset="", out=str(tmp_path / "new-report"),
        init_states="", grid=(288, 384), task=task_name, device="cuda:0",
        robot_usd="", teleop_device=device,
    )

    class EnvironmentBoundaryReached(Exception):
        pass

    def inspect_environment(_task, **kwargs):
        actual = kwargs["scene_cfg"].robot.spawn.articulation_props
        assert actual.solver_position_iteration_count == expected
        assert actual.solver_velocity_iteration_count == 1
        assert kwargs["xr_state_only"] is False
        raise EnvironmentBoundaryReached

    monkeypatch.setattr(module, "make_env", inspect_environment)
    with pytest.raises(EnvironmentBoundaryReached):
        module.Collector(args, task, knobs, app=None, teleop=teleop)


@pytest.mark.parametrize("value", ["0", "-1", "256", "nan", "inf", "-inf", "1.5", 0, -1, 256, float("nan"), float("inf"), 1.5, True])
def test_solver_iterations_reject_invalid_values_before_environment(monkeypatch, value):
    task = _load_blindpeg_task(monkeypatch)
    knob = task.knobs["robot_position_iterations"]
    with pytest.raises(ValueError, match="positive integer"):
        knob.type(value)
    knobs = {name: entry.default for name, entry in task.knobs.items()}
    knobs["robot_position_iterations"] = value
    with pytest.raises(ValueError, match="positive integer"):
        task.make_scene_cfg(knobs)


@pytest.mark.parametrize("count", [1, 12, 128, 255])
def test_solver_knob_cli_and_direct_scene_share_configuration(monkeypatch, count):
    task = _load_blindpeg_task(monkeypatch)
    parser = argparse.ArgumentParser()
    for name, knob in task.knobs.items():
        parser.add_argument(f"--{name}", type=knob.type, default=knob.default)
    knobs = vars(parser.parse_args(["--robot_position_iterations", str(count)]))
    cfg = task.make_scene_cfg(knobs)
    assert cfg.robot.spawn.articulation_props.solver_position_iteration_count == count
    assert cfg.robot.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert task.task_id(knobs) == "blind_peg_insertion"
