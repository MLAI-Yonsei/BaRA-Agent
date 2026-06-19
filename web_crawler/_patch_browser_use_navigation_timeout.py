"""Bump browser-use event timeouts for slow-egress hosts.

browser-use hardcodes per-event timeouts in browser_use/browser/events.py
(NavigateToUrlEvent=15s, ClickElementEvent=15s, BrowserStateRequestEvent=30s,
ScrollEvent=8s, TypeTextEvent=15s, etc.). On hosts where TCP connect to the
target takes 4–6s, those caps trigger before the page is interactable.

This patch raises the defaults for navigation, action, and DOM-state events
at import time. Per-class env vars override individual classes; otherwise all
covered classes use the value of BROWSER_USE_EVENT_TIMEOUT (default 180s).

Backward-compatible env var: BROWSER_USE_NAVIGATE_TIMEOUT, when set, applies
only to navigation-class events (preserves earlier semantics).
"""
import os


# Event classes we override. Grouped by category for readable env-var support.
_NAVIGATION_EVENTS = (
    "NavigateToUrlEvent",
    "NavigationStartedEvent",
    "NavigationCompleteEvent",
    "GoBackEvent",
    "GoForwardEvent",
    "RefreshEvent",
)

_ACTION_EVENTS = (
    "ClickElementEvent",
    "TypeTextEvent",
    "ScrollEvent",
    "ScrollToTextEvent",
    "SendKeysEvent",
    "WaitEvent",
)

_STATE_EVENTS = (
    "BrowserStateRequestEvent",
    "ScreenshotEvent",
    "SwitchTabEvent",
    "CloseTabEvent",
)


def _apply():
    try:
        from browser_use.browser import events as _ev
    except ImportError:
        return

    default_timeout = float(os.environ.get("BROWSER_USE_EVENT_TIMEOUT", "180"))
    nav_timeout = float(os.environ.get("BROWSER_USE_NAVIGATE_TIMEOUT", default_timeout))
    action_timeout = float(os.environ.get("BROWSER_USE_ACTION_TIMEOUT", default_timeout))
    state_timeout = float(os.environ.get("BROWSER_USE_STATE_TIMEOUT", default_timeout))

    bumped = []

    def _bump(class_names, timeout):
        for cls_name in class_names:
            cls = getattr(_ev, cls_name, None)
            if cls is None:
                continue
            if "event_timeout" in getattr(cls, "model_fields", {}):
                cls.model_fields["event_timeout"].default = timeout
                try:
                    cls.model_rebuild(force=True)
                except Exception:
                    pass
                bumped.append((cls_name, timeout))

    _bump(_NAVIGATION_EVENTS, nav_timeout)
    _bump(_ACTION_EVENTS, action_timeout)
    _bump(_STATE_EVENTS, state_timeout)

    if bumped:
        print(
            f"[patch] browser-use event timeouts bumped "
            f"(nav={nav_timeout}s, action={action_timeout}s, state={state_timeout}s; "
            f"{len(bumped)} class(es))"
        )


_apply()
