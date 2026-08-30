from __future__ import annotations

import math

from scripts.build_rotating_colloids_disorder_retention_summary import comparison


def test_matched_graph_comparison_reproduces_frozen_publication_values() -> None:
    rows = [
        {
            "graph_seeds": [17, 29, 43, 71, 97],
            "Q_target_conn": [0.4903, 0.5192, 0.4983, 0.5549, 0.5088],
        },
        {
            "graph_seeds": [17, 29, 43, 71, 97],
            "Q_target_conn": [0.4495, 0.4572, 0.4178, 0.4705, 0.4771],
        },
    ]

    result = comparison(rows)

    assert math.isclose(result["mean_difference_0p11_minus_0p16"], 0.05988, abs_tol=1e-12)
    assert math.isclose(result["paired_t"], 5.723572169356321, rel_tol=1e-12)
    assert math.isclose(result["paired_p_two_sided"], 0.004612046937705503, rel_tol=1e-12)
