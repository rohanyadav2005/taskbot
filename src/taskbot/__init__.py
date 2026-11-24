"""TaskBot: text-to-robot design with an LLM in the loop."""

__all__ = ["CheckResult"]


def __getattr__(name):
    # Lazy, so `import taskbot.platform_setup` works before MuJoCo is imported.
    if name == "CheckResult":
        from taskbot.checks import CheckResult

        return CheckResult
    raise AttributeError(f"module 'taskbot' has no attribute {name!r}")
