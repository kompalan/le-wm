"""Pixel-only MiniWoB++ environment for LeWM.

Wraps a MiniWoB++ task (Selenium + headless Chrome) as a gym env whose
observation is a square RGB frame and whose action is a single continuous
click position, so it plugs into the same data / planning stack as PushT.

- action: Box(2) in [-1, 1], mapped linearly onto the 160x210 task area
  (x right, y down). Every step is one CLICK_COORDS.
- pixels: the 210x160 screenshot, padded with white to 210x210 and resized
  to `img_size`. The instruction is drawn at the top of the page, so the
  goal is visible in pixels.

Tasks are restricted to multi-step click tasks: single-click tasks
(click-button, click-dialog, ...) terminate after one action, which is too
short for history_size + num_preds frames.
"""

import json

import gymnasium as gym
import miniwob
import numpy as np
from gymnasium import spaces
from miniwob.action import ActionSpaceConfig, ActionTypes
from PIL import Image

gym.register_envs(miniwob)

TASKS = ("click-checkboxes", "click-option", "click-collapsible", "click-tab-2")
TASK_W, TASK_H = 160, 210

# Interactive elements inside the task area, with the bits of page state that
# both the scripted experts and the success check need.
_PAGE_JS = """
const area = document.getElementById('area');
if (!area) return JSON.stringify({ready: false});
const box = e => { const r = e.getBoundingClientRect();
  return [r.left + r.width / 2, r.top + r.height / 2, r.width * r.height]; };
const vis = e => e.offsetParent !== null && box(e)[2] > 0;
const pick = (sel, f) => Array.from(area.querySelectorAll(sel)).filter(vis).map(f);
return JSON.stringify({
  ready: true,
  inputs: Array.from(area.querySelectorAll('input')).map(e => ({
    checked: e.checked, label: e.parentElement.textContent.trim(), c: box(e)})),
  submit: pick('#subbtn', e => ({c: box(e)})),
  headers: pick('.ui-accordion-header', e => ({
    expanded: e.getAttribute('aria-expanded') === 'true', c: box(e)})),
  tabs: pick('.ui-tabs-anchor', e => ({
    active: e.parentElement.classList.contains('ui-tabs-active'), c: box(e)})),
  links: pick('.alink', e => ({text: e.textContent.trim(), c: box(e)})),
});
"""


def to_frame(screenshot: np.ndarray, img_size: int) -> np.ndarray:
    """(210, 160, 3) screenshot -> white-padded square (img_size, img_size, 3)."""
    h, w, _ = screenshot.shape
    pad = (h - w) // 2
    sq = np.full((h, h, 3), 255, dtype=np.uint8)
    sq[:, pad : pad + w] = screenshot
    return np.asarray(Image.fromarray(sq).resize((img_size, img_size), Image.BILINEAR))


def norm_coords(xy) -> np.ndarray:
    """Task-area pixel coords -> action in [-1, 1]."""
    x, y = xy[0], xy[1]
    return np.array([2 * x / TASK_W - 1, 2 * y / TASK_H - 1], dtype=np.float32)


def raw_coords(action) -> np.ndarray:
    """Action in [-1, 1] -> task-area pixel coords (float32, as miniwob requires)."""
    a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
    return np.array([(a[0] + 1) / 2 * TASK_W, (a[1] + 1) / 2 * TASK_H], dtype=np.float32)


class MiniWoBClickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, task: str = "click-checkboxes", img_size: int = 224, render_mode: str = "rgb_array"):
        assert task in TASKS, f"unsupported task {task!r}, choose from {TASKS}"
        self.task = task
        self.img_size = img_size
        self.render_mode = render_mode
        self.env_name = f"MiniWoB/{task}"

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(img_size, img_size, 3), dtype=np.uint8)

        cfg = ActionSpaceConfig(action_types=[ActionTypes.CLICK_COORDS])
        self._env = gym.make(f"miniwob/{task}-v1", action_space_config=cfg)
        self._frame = None
        self._fields = {}
        self._visited_tabs = set()
        self.done = False

    @property
    def driver(self):
        return self._env.unwrapped.instance.driver

    # ---------------- Gym API ----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self._env.reset(seed=seed)
        # jQuery UI animates accordions/tabs; disable so the screenshot after a
        # click shows the settled page.
        self.driver.execute_script("if (window.jQuery) jQuery.fx.off = true;")
        self._fields = dict(obs["fields"])
        self._utterance = obs["utterance"]
        self._visited_tabs = set()
        self.done = False
        self._frame = to_frame(obs["screenshot"], self.img_size)
        return self._frame, self._get_info(reward=0.0)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        a = self._env.unwrapped.create_action(ActionTypes.CLICK_COORDS, coords=raw_coords(action))
        obs, reward, terminated, truncated, _ = self._env.step(a)
        self.done = terminated or truncated
        if not self.done:
            # after termination miniwob returns the blank post-task page
            self._frame = to_frame(obs["screenshot"], self.img_size)
        return self._frame, float(reward), terminated, truncated, self._get_info(reward=float(reward))

    def render(self):
        return self._frame

    def close(self):
        self._env.close()

    # ---------------- Page state ----------------

    def page(self) -> dict:
        return json.loads(self.driver.execute_script(_PAGE_JS))

    def page_state(self) -> str:
        """Canonical, comparable summary of the task-relevant DOM state."""
        if self.done:
            return "done"
        p = self.page()
        return json.dumps({
            "checked": [i["checked"] for i in p["inputs"]],
            "expanded": [h["expanded"] for h in p["headers"]],
            "tab": [t["active"] for t in p["tabs"]],
        })

    def _get_info(self, reward):
        return {
            "env_name": self.env_name,
            "pixels": self._frame,
            "utterance": self._utterance,
            "reward": reward,
            "done": self.done,
        }

    # ---------------- Scripted policies ----------------

    def expert_action(self):
        """Next expert click as (action, is_terminal). Terminal clicks end the task
        (Submit, or the target link in click-tab-2)."""
        p = self.page()
        if self.task in ("click-checkboxes", "click-option"):
            targets = {v for k, v in self._fields.items() if k.startswith("target")}
            for inp in p["inputs"]:
                if inp["checked"] != (inp["label"] in targets):
                    return norm_coords(inp["c"]), False
            return norm_coords(p["submit"][0]["c"]), True

        if self.task == "click-collapsible":
            if not p["headers"][0]["expanded"]:
                return norm_coords(p["headers"][0]["c"]), False
            return norm_coords(p["submit"][0]["c"]), True

        if self.task == "click-tab-2":
            for link in p["links"]:
                if link["text"] == self._fields["target"]:
                    return norm_coords(link["c"]), True
            for i, tab in enumerate(p["tabs"]):
                if tab["active"]:
                    self._visited_tabs.add(i)
            todo = [i for i in range(len(p["tabs"])) if i not in self._visited_tabs]
            i = todo[0] if todo else int(self.np_random.integers(len(p["tabs"])))
            return norm_coords(p["tabs"][i]["c"]), False

        raise NotImplementedError(self.task)

    def element_action(self, allow_terminal: bool = False):
        """Click the centre of a random interactive element (exploration)."""
        p = self.page()
        cands = [e["c"] for k in ("inputs", "headers", "tabs") for e in p[k]]
        if allow_terminal:
            cands += [e["c"] for k in ("submit", "links") for e in p[k]]
        if not cands:
            return self.action_space.sample()
        c = cands[int(self.np_random.integers(len(cands)))]
        jitter = self.np_random.normal(0, 2.0, size=2)  # a couple of pixels
        return np.clip(norm_coords(np.asarray(c[:2]) + jitter), -1, 1)


gym.register(id="swm/MiniWoB-v0", entry_point="miniwob_env:MiniWoBClickEnv")
