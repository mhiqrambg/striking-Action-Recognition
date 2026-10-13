from src.realtime.early_warning import EarlyWarning

__all__ = ["EarlyWarning", "RealtimePredictor"]


def __getattr__(name):
    # [lazy import] RealtimePredictor menarik torch/ultralytics; impor secara
    # lazy supaya `from src.realtime import EarlyWarning` tetap ringan.
    if name == "RealtimePredictor":
        from src.realtime.predictor import RealtimePredictor
        return RealtimePredictor
    raise AttributeError(name)
