import pytest

from scripts.plot_rotating_colloids_activated_memory_prl import first_crossing_time


def test_interpolated_crossing_and_censoring():
    assert first_crossing_time([0, 1, 2], [1, 0.6, 0.4]) == pytest.approx(1.5)
    assert first_crossing_time([0, 1, 2], [1, 0.8, 0.7]) is None


def test_first_crossing_is_not_replaced_by_later_recrossing():
    assert first_crossing_time([0, 1, 2, 3], [1, 0.4, 0.7, 0.2]) == pytest.approx(5 / 6)


def test_invalid_time_order_rejected():
    with pytest.raises(ValueError):
        first_crossing_time([0, 2, 1], [1, 0.8, 0.4])
