"""browser-use 0.7.x passes top_p/seed into AsyncOpenAI(); the OpenAI SDK rejects them."""

from __future__ import annotations


def apply() -> None:
    try:
        from browser_use.llm.openrouter import chat as openrouter_chat
    except ImportError:
        return

    cls = openrouter_chat.ChatOpenRouter
    if getattr(cls, "_bara_async_openai_client_patch", False):
        return

    _orig = cls._get_client_params

    def _get_client_params(self):  # type: ignore[no-untyped-def]
        params = _orig(self)
        params.pop("top_p", None)
        params.pop("seed", None)
        return params

    cls._get_client_params = _get_client_params  # type: ignore[method-assign]
    cls._bara_async_openai_client_patch = True


apply()
