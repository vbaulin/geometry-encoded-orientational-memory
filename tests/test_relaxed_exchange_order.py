from scripts.test_relaxed_exchange_order import run_case


def test_large_write_bracket_does_not_guarantee_retention():
    partial = run_case(4, 0.30)
    full = run_case(8, 0.30)
    assert full["bracket_norm"] > partial["bracket_norm"]
    assert full["separation_after_write"] > 1.69
    assert full["retained"] is False
    assert partial["retained"] is True


def test_partial_contest_transmits_order_outside_write_support():
    partial = run_case(4, 0.30)
    assert partial["retained_coordinates"] == 8
    assert partial["retained_outside_common_support"] == 4


def test_full_support_null_is_coupling_dependent():
    assert run_case(8, 0.30)["retained"] is False
    assert run_case(8, 0.13)["retained_coordinates"] == 4
