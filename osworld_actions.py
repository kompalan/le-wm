"""Parse OSWorld agent actions into one schema, and encode them as fixed vectors.

Agents in xlangai/ubuntu_osworld_verified_trajs log actions in four formats:

  claude      {"tool_call": {"name": "computer", "input": {"action": "left_click", "coordinate": [x, y]}}}
              ("batched_actions" holds a list of these)
  pyautogui   "import pyautogui; pyautogui.click(150, 431); pyautogui.write('Total')"
              (UI-TARS, Agent S, Gemini, ...; also wrapped as {"action_space": "pyautogui", "action": "..."})
  computer13  {"action_type": "CLICK", "x": 1100, "y": 146}  (OSWorld's own action space,
              plus AutoGLM's OPEN_APP)
  special     "WAIT", "DONE", "FAIL"

parse_step() turns one logged step into a list of canonical sub-actions:

  {"type": <TYPES>, "x", "y"                 # pointer target in [-1, 1] (screen-relative)
   "x2", "y2"                                # drag end
   "dx", "dy"                                # scroll, in wheel clicks (dy > 0 = down)
   "keys": [["ctrl", "a"], ["enter"]]        # key combos pressed in order
   "text": "Zurich"}                         # typed text (app name for open_app)

or None for terminal / unparseable steps. Coordinates are pixels divided by the
screenshot size, so agents with different screen sizes share one space.
encode_step() maps the sub-action list to a fixed float vector (ACTION_DIM).
"""

import ast
import json
import math
import re
import zlib

import numpy as np

TYPES = ["noop", "move", "click", "double_click", "triple_click", "right_click", "middle_click",
         "drag", "scroll", "key", "type", "mouse_down", "mouse_up", "key_down", "key_up", "open_app"]
MODIFIERS = ["ctrl", "shift", "alt", "super"]
MAX_SUB = 4  # sub-actions encoded per step; the rest are dropped (counted as truncated)
HASH_DIM = 16

_KEY_ALIASES = {
    "return": "enter", "control": "ctrl", "ctrlleft": "ctrl", "ctrlright": "ctrl", "control_l": "ctrl",
    "control_r": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl", "shiftleft": "shift", "shiftright": "shift",
    "shift_l": "shift", "shift_r": "shift", "altleft": "alt", "altright": "alt", "alt_l": "alt", "alt_r": "alt",
    "option": "alt", "cmd": "super", "command": "super", "win": "super", "winleft": "super", "meta": "super",
    "super_l": "super", "super_r": "super", "escape": "esc", "page_down": "pagedown", "page_up": "pageup",
    "next": "pagedown", "prior": "pageup", "del": "delete", "backspace": "backspace", "kp_enter": "enter",
}


def norm_key(k):
    k = str(k).strip().lower().replace(" ", "")
    return _KEY_ALIASES.get(k, k)


def _combo(s):
    """'ctrl+shift+t' -> ['ctrl', 'shift', 't'] ('+' alone stays a key)."""
    s = str(s)
    if s.strip() == "+":
        return ["+"]
    return [norm_key(p) for p in s.split("+") if p != ""]


class _Pointer:
    """Pointer position in [-1, 1]; pyautogui calls without coordinates act at it."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.x = self.y = None

    def norm(self, x, y):
        return 2 * float(x) / self.w - 1, 2 * float(y) / self.h - 1

    def at(self, x=None, y=None):
        if x is not None and y is not None:
            self.x, self.y = self.norm(x, y)
        return self.x, self.y


def _act(type_, **kw):
    return {"type": type_, **{k: v for k, v in kw.items() if v is not None}}


def _pointer_act(type_, ptr, x=None, y=None):
    px, py = ptr.at(x, y)
    return _act(type_, x=px, y=py)


# ---------------- claude computer-use tool ----------------

_CLAUDE_CLICKS = {"left_click": "click", "double_click": "double_click", "triple_click": "triple_click",
                  "right_click": "right_click", "middle_click": "middle_click", "mouse_move": "move",
                  "left_mouse_down": "mouse_down", "left_mouse_up": "mouse_up"}
_CLAUDE_NOOPS = {"screenshot", "zoom", "wait", "cursor_position"}
_SCROLL_DIRS = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}


def _claude(inp, ptr):
    a = inp.get("action")
    c = inp.get("coordinate")
    if a == "batched_actions":
        out = []
        for sub in inp.get("actions", []):
            out += _claude(sub, ptr) or []
        return out
    if a in _CLAUDE_CLICKS:
        acts = []
        if inp.get("text"):  # modifier held during the click, e.g. "shift"
            acts.append(_act("key_down", keys=[_combo(inp["text"])]))
        acts.append(_pointer_act(_CLAUDE_CLICKS[a], ptr, *(c or (None, None))))
        if inp.get("text"):
            acts.append(_act("key_up", keys=[_combo(inp["text"])]))
        return acts
    if a == "left_click_drag":
        s = inp.get("start_coordinate")
        x0, y0 = ptr.at(*(s or (None, None)))
        x1, y1 = ptr.at(*c)
        return [_act("drag", x=x0, y=y0, x2=x1, y2=y1)]
    if a == "scroll":
        sx, sy = _SCROLL_DIRS.get(inp.get("scroll_direction", "down"), (0, 1))
        n = float(inp.get("scroll_amount", 1))
        x, y = ptr.at(*(c or (None, None)))
        return [_act("scroll", x=x, y=y, dx=sx * n, dy=sy * n)]
    if a in ("key", "hold_key"):
        return [_act("key", keys=[_combo(k) for k in str(inp.get("text", "")).split()])]
    if a == "type":
        return [_act("type", text=str(inp.get("text", "")))]
    if a in _CLAUDE_NOOPS:
        return [_act("noop")]
    raise ValueError(f"unknown claude action {a!r}")


# ---------------- pyautogui code ----------------

_PG_CLICKS = {"click": "click", "leftClick": "click", "doubleClick": "double_click", "tripleClick": "triple_click",
              "rightClick": "right_click", "middleClick": "middle_click", "moveTo": "move",
              "mouseDown": "mouse_down", "mouseUp": "mouse_up"}
_PG_IGNORE = {"sleep", "PAUSE", "FAILSAFE", "size", "position", "screenshot", "locateOnScreen", "print"}


def _lit(node):
    return ast.literal_eval(node)


def _xy(args, kw):
    """pyautogui's x/y: positional (x, y), a tuple (x, y), or keywords."""
    x = kw.get("x", args[0] if len(args) > 0 else None)
    y = kw.get("y", args[1] if len(args) > 1 else None)
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        x, y = x[0], x[1]
    return x, y


def _pg_call(name, args, kw, ptr):
    if name in _PG_CLICKS:
        x, y = _xy(args, kw)
        t = _PG_CLICKS[name]
        if name == "click":
            n, button = int(kw.get("clicks", 1)), kw.get("button", "left")
            t = {"right": "right_click", "middle": "middle_click"}.get(button,
                {1: "click", 2: "double_click", 3: "triple_click"}.get(n, "click"))
        elif name in ("mouseDown", "mouseUp") and kw.get("button", "left") != "left":
            pass  # rare; kept as left
        return [_pointer_act(t, ptr, x, y)]
    if name in ("moveRel", "move"):
        dx, dy = _xy(args, kw)
        if ptr.x is None:
            return [_act("move")]
        return [_act("move", x=ptr.x + 2 * float(dx or 0) / ptr.w, y=ptr.y + 2 * float(dy or 0) / ptr.h)]
    if name in ("dragTo", "drag", "dragRel"):
        x0, y0 = ptr.x, ptr.y
        x, y = _xy(args, kw)
        if name != "dragTo":
            if x0 is None:
                return [_act("drag")]
            x1, y1 = x0 + 2 * float(x or 0) / ptr.w, y0 + 2 * float(y or 0) / ptr.h
            ptr.x, ptr.y = x1, y1
        else:
            x1, y1 = ptr.at(x, y)
        return [_act("drag", x=x0, y=y0, x2=x1, y2=y1)]
    if name in ("scroll", "vscroll", "hscroll"):
        n = float(kw.get("clicks", args[0] if args else 0))
        x, y = ptr.at(kw.get("x", args[1] if len(args) > 1 else None), kw.get("y", args[2] if len(args) > 2 else None))
        # pyautogui: positive clicks scroll up (vscroll) / right (hscroll)
        dx, dy = (n, 0.0) if name == "hscroll" else (0.0, -n)
        return [_act("scroll", x=x, y=y, dx=dx, dy=dy)]
    if name in ("write", "typewrite"):
        msg = kw.get("message", args[0] if args else "")
        if isinstance(msg, (list, tuple)):  # typewrite(['a', 'enter']) presses keys
            return [_act("key", keys=[[norm_key(k)] for k in msg])]
        return [_act("type", text=str(msg))]
    if name == "press":
        keys = kw.get("keys", args[0] if args else "")
        keys = list(keys) if isinstance(keys, (list, tuple)) else [keys]
        n = int(kw.get("presses", 1))
        return [_act("key", keys=[[norm_key(k)] for k in keys] * n)]
    if name == "hotkey":
        keys = list(args) if args else list(kw.get("keys", []))
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = list(keys[0])
        return [_act("key", keys=[[norm_key(k) for k in keys]])]
    if name in ("keyDown", "keyUp"):
        k = kw.get("key", args[0] if args else "")
        return [_act("key_down" if name == "keyDown" else "key_up", keys=[[norm_key(k)]])]
    if name in _PG_IGNORE:
        return []
    raise ValueError(f"unknown pyautogui call {name!r}")


def _pyautogui(code, ptr):
    s = code.strip()
    if s in ("WAIT",):
        return [_act("noop")]
    if s in ("DONE", "FAIL", ""):
        return None
    tree = ast.parse(s)
    out = []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    clipboard = None
    for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):  # source order
        mod = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        if mod not in ("pyautogui", "time", "pyperclip"):
            continue
        args = [_lit(a) for a in node.args]
        kw = {k.arg: _lit(k.value) for k in node.keywords if k.arg}
        if mod == "pyperclip":
            # UI-TARS types via the clipboard: pyperclip.copy(text); pyautogui.hotkey('ctrl', 'v')
            if node.func.attr == "copy":
                clipboard = str(kw.get("text", args[0] if args else ""))
            continue
        acts = _pg_call(node.func.attr, args, kw, ptr)
        if clipboard is not None and acts and acts[0].get("keys") == [["ctrl", "v"]]:
            acts = [_act("type", text=clipboard)]
        out += acts
    return out or [_act("noop")]


# ---------------- OSWorld computer_13 ----------------

def _computer13(a, ptr):
    t = a.get("action_type", "").upper()
    p = a.get("parameters", a)
    x, y = p.get("x"), p.get("y")
    if t in ("DONE", "FAIL"):
        return None
    if t == "WAIT":
        return [_act("noop")]
    if t in ("CLICK",):
        n, button = int(p.get("num_clicks", 1)), str(p.get("button", "left")).lower()
        tt = {"right": "right_click", "middle": "middle_click"}.get(button,
             {1: "click", 2: "double_click", 3: "triple_click"}.get(n, "click"))
        return [_pointer_act(tt, ptr, x, y)]
    simple = {"MOVE_TO": "move", "RIGHT_CLICK": "right_click", "DOUBLE_CLICK": "double_click",
              "MOUSE_DOWN": "mouse_down", "MOUSE_UP": "mouse_up"}
    if t in simple:
        return [_pointer_act(simple[t], ptr, x, y)]
    if t == "DRAG_TO":
        x0, y0 = ptr.x, ptr.y
        x1, y1 = ptr.at(x, y)
        return [_act("drag", x=x0, y=y0, x2=x1, y2=y1)]
    if t == "SCROLL":
        # computer_13 follows pyautogui: positive dy scrolls up
        return [_act("scroll", x=ptr.x, y=ptr.y, dx=float(p.get("dx", 0)), dy=-float(p.get("dy", 0)))]
    if t == "TYPING":
        return [_act("type", text=str(p.get("text", "")))]
    if t == "PRESS":
        return [_act("key", keys=[[norm_key(p.get("key", ""))]])]
    if t == "HOTKEY":
        return [_act("key", keys=[[norm_key(k) for k in p.get("keys", [])]])]
    if t in ("KEY_DOWN", "KEY_UP"):
        return [_act("key_down" if t == "KEY_DOWN" else "key_up", keys=[[norm_key(p.get("key", ""))]])]
    if t == "OPEN_APP":  # AutoGLM: launch an application by name
        return [_act("open_app", text=str(p.get("app_name", p.get("launch_app_command", ""))))]
    raise ValueError(f"unknown computer_13 action {t!r}")


def parse_step(step, ptr):
    """One traj.jsonl record -> list of canonical sub-actions, or None (terminal step).
    Raises ValueError/SyntaxError on actions it cannot parse."""
    if "tool_call" in step:
        tc = step["tool_call"]
        if not tc:
            return None  # final answer without an action
        return _claude(tc.get("input", {}), ptr)
    a = step.get("action")
    if isinstance(a, dict):
        if "action_type" in a:
            return _computer13(a, ptr)
        if "action" in a:
            a = a["action"]
    if isinstance(a, list):  # some agents log a list of code strings / dicts
        out = []
        for sub in a:
            r = parse_step({"action": sub}, ptr)
            if r is None:
                return None if not out else out
            out += r
        return out
    if a is None:
        return None
    if isinstance(a, str):
        return _pyautogui(a, ptr)
    raise ValueError(f"unrecognised action record {str(a)[:80]!r}")


def pointer(w, h):
    return _Pointer(w, h)


# ---------------- encoding ----------------

def _hash_bag(items, dim=HASH_DIM):
    v = np.zeros(dim, np.float32)
    for it in items:
        h = zlib.crc32(it.encode("utf-8", "ignore"))
        v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


SUB_DIM = len(TYPES) + 5 + 2 + len(MODIFIERS) + HASH_DIM + 1 + 2 + HASH_DIM
ACTION_DIM = MAX_SUB * SUB_DIM + 1


def encode_sub(a):
    v = np.zeros(SUB_DIM, np.float32)
    i = 0
    v[TYPES.index(a["type"])] = 1.0
    i += len(TYPES)
    if a.get("x") is not None:
        v[i:i + 2] = a["x"], a["y"]
        v[i + 4] = 1.0
    if a.get("x2") is not None:
        v[i + 2:i + 4] = a["x2"], a["y2"]
    i += 5
    v[i:i + 2] = a.get("dx", 0.0) / 10.0, a.get("dy", 0.0) / 10.0
    i += 2
    combos = a.get("keys", [])
    flat = [k for c in combos for k in c]
    for j, m in enumerate(MODIFIERS):
        v[i + j] = float(m in flat)
    i += len(MODIFIERS)
    v[i:i + HASH_DIM] = _hash_bag([k for k in flat if k not in MODIFIERS])
    i += HASH_DIM
    v[i] = math.log1p(len(combos)) / 3.0
    i += 1
    text = a.get("text", "")
    v[i:i + 2] = math.log1p(len(text)) / 5.0, float("\n" in text)
    i += 2
    t = text.lower()
    v[i:i + HASH_DIM] = _hash_bag([t[k:k + 2] for k in range(max(0, len(t) - 1))] or list(t))
    return v


def encode_step(subs):
    """Sub-action list -> (ACTION_DIM,) vector: MAX_SUB slots (empty slots are 0) + sub-action count.
    No-ops (waits, screenshots) inside a multi-action step are dropped first."""
    subs = [a for a in subs if a["type"] != "noop"] or subs[:1]
    v = np.zeros(ACTION_DIM, np.float32)
    for j, a in enumerate(subs[:MAX_SUB]):
        v[j * SUB_DIM:(j + 1) * SUB_DIM] = encode_sub(a)
    v[-1] = math.log1p(len(subs)) / 2.0
    return v


def cursor_after(subs, prev):
    """Pointer position after the step (None if never known)."""
    pos = prev
    for a in subs:
        if a.get("x2") is not None:
            pos = (a["x2"], a["y2"])
        elif a.get("x") is not None and a["type"] != "scroll":
            pos = (a["x"], a["y"])
        elif a["type"] == "scroll" and a.get("x") is not None:
            pos = (a["x"], a["y"])
    return pos


def dumps(subs):
    return json.dumps(subs, separators=(",", ":"))
