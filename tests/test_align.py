from shelftwin.planogram import (AlignConfig, ExpectedSlot, ObservedPack, align_row, bay_layout_changed,
                                 draft_from_observations, slot_verdicts)

EXP = [ExpectedSlot("p1", 10, "A"), ExpectedSlot("p2", 30, "B"), ExpectedSlot("p3", 50, "C")]


def kinds(out):
    return [o["kind"] for o in out]


def test_all_match():
    obs = [ObservedPack(11, "A"), ObservedPack(29, "B"), ObservedPack(51, "C")]
    assert kinds(align_row(EXP, obs)) == ["match"] * 3


def test_gap_detected():
    obs = [ObservedPack(11, "A"), ObservedPack(51, "C")]
    out = align_row(EXP, obs)
    assert slot_verdicts(out) == {"p1": "match", "p2": "gap", "p3": "match"}


def test_substitution_is_displacement_not_corrected():
    obs = [ObservedPack(11, "A"), ObservedPack(30, "C"), ObservedPack(51, "C")]
    assert slot_verdicts(align_row(EXP, obs))["p2"] == "substitution"


def test_unknown_identity_is_uncertain():
    obs = [ObservedPack(11, "A"), ObservedPack(30, "unknown"), ObservedPack(51, "C")]
    assert slot_verdicts(align_row(EXP, obs))["p2"] == "uncertain"


def test_insertion():
    obs = [ObservedPack(11, "A"), ObservedPack(20, "X"), ObservedPack(29, "B"), ObservedPack(51, "C")]
    out = align_row(EXP, obs)
    assert "insertion" in kinds(out)
    assert slot_verdicts(out) == {"p1": "match", "p2": "match", "p3": "match"}


def test_far_position_prefers_gap_plus_insertion():
    # A pack of 'A' observed far away (100 cm) should not be matched to slot p1 at 10 cm
    obs = [ObservedPack(100, "A")]
    out = align_row([ExpectedSlot("p1", 10, "A")], obs, AlignConfig())
    assert sorted(kinds(out)) == ["gap", "insertion"]


def test_layout_changed_guard():
    rows = [[dict(kind="substitution", slot_id=f"s{i}") for i in range(5)] + [dict(kind="match", slot_id="m")]]
    assert bay_layout_changed(rows)
    rows = [[dict(kind="substitution", slot_id="s")] + [dict(kind="match", slot_id=f"m{i}") for i in range(5)]]
    assert not bay_layout_changed(rows)


def test_draft_merges_facings():
    lay = draft_from_observations("B1", {1: [(0, 10, "A"), (10, 10, "A"), (20, 10, "B")]})
    assert [(s.sku_id, s.facings) for s in lay.slots] == [("A", 2), ("B", 1)]
    assert lay.status == "draft"
