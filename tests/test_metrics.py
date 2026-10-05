from app.evaluation.metrics import aggregate, cer, wer


def test_cer_and_wer_on_ottoman_words():
    assert cer("قرية", "قريه") == 0.25
    assert wer("قرية فلان", "قرية فلان") == 0.0
    assert wer("a b", "a c") == 0.5


def test_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0
    assert wer("", "") == 0.0


def test_aggregate_micro_average():
    scores = aggregate([("aa", "ab"), ("bbbb", "bbbb")])
    assert scores["lines"] == 2
    assert scores["cer"] == 1 / 6
    assert scores["wer"] == 0.5
