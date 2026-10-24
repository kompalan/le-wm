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
from miniwob.screenshot import get_screenshot, pil_to_numpy_array
from PIL import Image, ImageDraw

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


# Diagnostic variant: make all text transparent (instruction, labels, tab
# titles, links) while keeping the layout and widget state visible.
# Replace the episode's 10 s timer with one that never fires in practice (~11
# days). EPISODE_MAX_TIME also scales time-proportional rewards to ~1.
_NO_TIMEOUT_JS = """
core.EPISODE_MAX_TIME = 1e9;
if (core.EP_TIMER !== null) {
  clearTimeout(core.EP_TIMER);
  core.EP_TIMER = setTimeout(function () { core.endEpisode(-1, false, 'timed out'); }, 1e9);
}
"""

_HIDE_TEXT_JS = """
if (!document.getElementById('hide-text')) {
  const s = document.createElement('style');
  s.id = 'hide-text';
  s.textContent = '* { color: transparent !important; text-shadow: none !important; }';
  document.head.appendChild(s);
}
"""


# Mouse pointer drawn at the last click (draw_cursor=True), tip at (0, 0),
# in task-area pixels; roughly a standard 12 px arrow.
_ARROW = [(0, 0), (0, 12), (3, 9), (5, 14), (7, 13), (5, 8), (9, 8)]


def draw_arrow(screenshot: np.ndarray, xy) -> np.ndarray:
    img = Image.fromarray(screenshot)
    x, y = float(xy[0]), float(xy[1])
    ImageDraw.Draw(img).polygon([(x + dx, y + dy) for dx, dy in _ARROW], fill="black", outline="white")
    return np.asarray(img)


def page_changed(a: np.ndarray, b: np.ndarray, thr: int = 30, min_px: int = 10) -> bool:
    """>min_px pixels differ by >thr in some channel (same test as the probes)."""
    return bool(((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > thr).sum()) > min_px)


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


def frame_coords(action, img_size: int) -> np.ndarray:
    """Action in [-1, 1] -> (x, y) pixel in the square observation frame."""
    x, y = raw_coords(action)
    pad = (TASK_H - TASK_W) // 2  # to_frame pads the width to a square
    return np.array([(x + pad), y]) * img_size / TASK_H


class MiniWoBClickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, task: str = "click-checkboxes", img_size: int = 224, render_mode: str = "rgb_array",
                 hide_text: bool = False, draw_cursor: bool = False):
        assert task in TASKS, f"unsupported task {task!r}, choose from {TASKS}"
        self.task = task
        self.hide_text = hide_text
        self.draw_cursor = draw_cursor  # observations show a pointer at the last click
        self._cursor = None
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
        # MiniWoB fails an episode 10 s of wall-clock time after reset, so a slow
        # planner (or a loaded machine) fails regardless of its clicks
        self.driver.execute_script(_NO_TIMEOUT_JS)
        self._fields = dict(obs["fields"])
        self._utterance = obs["utterance"]
        self._visited_tabs = set()
        self.done = False
        screenshot = obs["screenshot"]
        if self.hide_text:
            self.driver.execute_script(_HIDE_TEXT_JS)
            screenshot = self._screenshot()  # the reset observation predates the style
        self._cursor = None  # no click yet, so no pointer
        self._set_frames(screenshot)
        return self._frame, self._get_info(reward=0.0)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        coords = raw_coords(action)
        a = self._env.unwrapped.create_action(ActionTypes.CLICK_COORDS, coords=coords)
        obs, reward, terminated, truncated, _ = self._env.step(a)
        self.done = terminated or truncated
        if not self.done:
            # after termination miniwob returns the blank post-task page
            self._cursor = coords
            self._set_frames(obs["screenshot"])
        return self._frame, float(reward), terminated, truncated, self._get_info(reward=float(reward))

    def _set_frames(self, screenshot):
        self.clean_frame = to_frame(screenshot, self.img_size)  # page only, never a pointer
        if self.draw_cursor and self._cursor is not None:
            screenshot = draw_arrow(screenshot, self._cursor)
        self._frame = to_frame(screenshot, self.img_size)

    def _screenshot(self):
        inst = self._env.unwrapped.instance
        img = get_screenshot(self.driver, inst.inner_width, inst.inner_height, inst.task_width, inst.task_height)
        return pil_to_numpy_array(img)

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
