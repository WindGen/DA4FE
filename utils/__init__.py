__all__ = [
    "CVPRPlotStyle",
    "list_available_metrics",
    "load_training_log",
    "plot_training_curves",
]


def __getattr__(name):
    if name in __all__:
        from .log_plotting import (
            CVPRPlotStyle,
            list_available_metrics,
            load_training_log,
            plot_training_curves,
        )

        exports = {
            "CVPRPlotStyle": CVPRPlotStyle,
            "list_available_metrics": list_available_metrics,
            "load_training_log": load_training_log,
            "plot_training_curves": plot_training_curves,
        }
        return exports[name]
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
