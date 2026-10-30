"""RoboCasa 폐루프 평가 클라이언트 — GR00T N1.6 정책 서버(gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper)용.

GR00T N1.5 의 RoboCasa 평가 프로토콜을 그대로 따른다 (결과를 N1.5 와 직접 비교하기 위해):
robocasa gym env(load_robocasa_gym_env + RoboCasaWrapper, 같은 관측 키), generative textures "100p",
obj_instance_split A, styles [0..8,11], layout -1, seed 42, RecordVideo, MultiStepWrapper
(n_action_steps = 실행 길이), 이미지 세로 뒤집기, 결과는 prediction.txt 에
"episode i is_success: [..] action_steps: N" 한 줄씩. 이미 기록된 에피소드는 건너뛴다
(건너뛰어도 env.reset() 은 불러 장면 순서가 바뀌지 않게 한다). 최대 스텝 = env horizon.

통신(N1.6 PolicyServer, msgpack + np.save 배열), Gr00tSimPolicyWrapper 용 평평한 키:
  video.<k>: uint8 (B=1, T=1, H, W, 3); state.<k>: float32 (1, 1, D);
  annotation.human.action.task_description: [str]  ->  action.<k>: float32 (1, 16, D).
torch 가 필요 없다. robosuite/robocasa 가 깔린 평가 env 에서 돈다(README 참고).
"""
import argparse
import io
import json
import os
import re
from pathlib import Path

import msgpack
import numpy as np
import zmq
from tqdm import tqdm, trange

from wrappers.multistep_wrapper import MultiStepWrapper
from wrappers.record_video import RecordVideo
from wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env


def _enc(obj):
    if isinstance(obj, np.ndarray):
        b = io.BytesIO(); np.save(b, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": b.getvalue()}
    return obj


def _dec(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if isinstance(obj, dict) and "__ModalityConfig_class__" in obj:
        return obj["as_json"]
    return obj


class PolicyClient:
    def __init__(self, host, port, timeout_ms=600000):
        self.ctx = zmq.Context(); self.addr = f"tcp://{host}:{port}"; self.timeout = timeout_ms
        self._sock()

    def _sock(self):
        self.s = self.ctx.socket(zmq.REQ)
        self.s.setsockopt(zmq.RCVTIMEO, self.timeout); self.s.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.s.connect(self.addr)

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.s.send(msgpack.packb(req, default=_enc))
        resp = msgpack.unpackb(self.s.recv(), object_hook=_dec)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def get_action(self, obs):
        action, _info = self.call("get_action", {"observation": obs, "options": None})
        return action


def done_episodes(pred):
    if not os.path.exists(pred):
        return set()
    return {int(m.group(1)) for m in re.finditer(r"^episode\s+(\d+)\s+is_success:", open(pred).read(), re.M)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost"); p.add_argument("--port", type=int, default=5555)
    p.add_argument("--env_name", required=True); p.add_argument("--video_dir", required=True)
    p.add_argument("--n_episodes", type=int, default=50); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_horizon", type=int, default=16, help="executed steps per chunk (EXEC_H)")
    p.add_argument("--max_episode_steps", type=int, default=1500, help="progress bar only (env horizon ends episodes)")
    p.add_argument("--robots", default="PandaOmron")
    p.add_argument("--layout", type=int, nargs="+", default=-1)
    p.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    p.add_argument("--generative_textures", action="store_true")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--dump_obs", default=None, help="save the first (flipped, batched) observation sent to the server as .npz")
    a = p.parse_args()

    client = PolicyClient(a.host, a.port)
    print("ping:", client.call("ping"))
    mc = client.call("get_modality_config")
    vkeys = mc["video"]["modality_keys"]; skeys = mc["state"]["modality_keys"]
    lkey = mc["language"]["modality_keys"][0]; akeys = mc["action"]["modality_keys"]
    horizon = len(mc["action"]["delta_indices"])
    print(f"modality: video={vkeys} state={skeys} lang={lkey} action={akeys} horizon={horizon}")
    assert a.action_horizon <= horizon, f"EXEC_H {a.action_horizon} > model chunk {horizon}"

    env = load_robocasa_gym_env(
        a.env_name, seed=a.seed, robots=a.robots, camera_widths=256, camera_heights=256,
        render_onscreen=False, obj_instance_split="A",
        generative_textures="100p" if a.generative_textures else None, randomize_cameras=False,
        layout_ids=a.layout, style_ids=a.style, collect_data=False)
    env = RoboCasaWrapper(env)
    os.makedirs(a.video_dir, exist_ok=True)
    pred = os.path.join(a.video_dir, "prediction.txt")
    done = done_episodes(pred)
    print(f"already done: {sorted(done)}")
    if not a.no_video:
        trigger = len(done) if done else 1
        env = RecordVideo(env, Path(a.video_dir), disable_logger=True,
                          episode_trigger=lambda t: t % trigger == 0, fps=20, name_prefix=a.env_name)
    env = MultiStepWrapper(env, video_delta_indices=np.arange(1), state_delta_indices=np.arange(1),
                           n_action_steps=a.action_horizon)

    def to_wire(obs):
        out = {}
        for k in vkeys:  # same vertical flip as N1.5 (axis 1 of (T,H,W,C) = H), then batch dim
            out[f"video.{k}"] = np.ascontiguousarray(np.flip(obs[f"video.{k}"], axis=1)[None]).astype(np.uint8)
        for k in skeys:
            out[f"state.{k}"] = np.asarray(obs[f"state.{k}"], dtype=np.float32)[None]
        out[lkey] = [str(obs[lkey][0])]
        return out

    logged = False
    for i in trange(a.n_episodes):
        obs, info = env.reset()
        if i in done:
            print(f"Skipping episode {i} as it has already been evaluated.")
            continue
        pbar = tqdm(total=a.max_episode_steps, desc=f"Episode {i} / {env.unwrapped.get_ep_meta()['lang']}", leave=False)
        fin = False; step = 0
        while not fin:
            wire = to_wire(obs)
            if a.dump_obs and not os.path.exists(a.dump_obs):
                np.savez(a.dump_obs, **{k: (np.array(v) if k == lkey else v) for k, v in wire.items()})
            act = client.get_action(wire)
            act = {k: np.asarray(v)[0] for k, v in act.items()}  # drop batch -> (16, D)
            if not logged:
                print("[wire] obs " + json.dumps({k: [str(v.dtype), list(v.shape)] if isinstance(v, np.ndarray) else v
                                                  for k, v in wire.items()}))
                print("[act] " + json.dumps({k: list(v.shape) for k, v in act.items()})
                      + f" total_dim={sum(v.shape[-1] for v in act.values())} exec={a.action_horizon}")
                print("[act0] " + json.dumps({k: np.round(v[0], 4).tolist() for k, v in act.items()}))
                logged = True
            obs, reward, terminated, truncated, info = env.step(act)
            fin = terminated or truncated
            step += a.action_horizon
            pbar.update(a.action_horizon)
        pbar.close()
        with open(pred, "a") as f:
            f.write(f"episode {i} is_success: {info['is_success']} action_steps: {step}\n")
    env.close()
    n = done_episodes(pred)
    print(json.dumps({"task": a.env_name, "episodes_done": len(n)}))
