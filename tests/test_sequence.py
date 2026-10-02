import pytest

from mpme.sequence import Protocol, Scan, mpme_echo_times, paper_protocol


def test_fig1_echo_order_alternates():
    # Window 1 visits +1, 0, −1; window 2 reverses; window 3 repeats window 1.
    t = dict(zip((1, 0, -1), mpme_echo_times((1, 0, -1), 3, 2.0, 4.5)))
    assert t[1] == (4.5, 14.5, 16.5)
    assert t[0] == (6.5, 12.5, 18.5)
    assert t[-1] == (8.5, 10.5, 20.5)


def test_paper_protocol_values():
    pr = paper_protocol()
    assert [s.flip_deg for s in pr.scans] == [15.0, 330.0]
    assert [s.TR for s in pr.scans] == [25.0, 25.0]
    assert pr.pathways == (1, 0, -1) and pr.n_echoes == 3


def test_protocol_validation():
    with pytest.raises(ValueError):
        Protocol((0, -1), (Scan(15.0, 25.0, ((5.0, 30.0), (6.0, 7.0))),))  # TE > TR
    with pytest.raises(ValueError):
        Protocol((0, -1), (Scan(15.0, 25.0, ((5.0, 6.0), (7.0,))),))       # ragged
