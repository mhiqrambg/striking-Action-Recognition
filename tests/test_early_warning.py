from src.realtime.early_warning import EarlyWarning


def test_triggers_after_5_consecutive():
    ew = EarlyWarning(threshold=5)
    results = [ew.update("light") for _ in range(5)]
    assert results[-1] is True
    assert results[0] is False


def test_resets_on_normal():
    ew = EarlyWarning(threshold=5)
    for _ in range(4):
        ew.update("heavy")
    ew.update("normal")
    assert ew.update("heavy") is False


def test_not_trigger_for_normal():
    ew = EarlyWarning(threshold=5)
    assert all(ew.update("normal") is False for _ in range(10))
