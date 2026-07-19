from util.splits import build_train_holdout_indices


def test_count_stratified_holdout_is_deterministic_and_count_representative():
    counts = list(range(100))
    first = build_train_holdout_indices(
        len(counts), 0.2, 17, strategy='count_stratified', sample_counts=counts
    )
    second = build_train_holdout_indices(
        len(counts), 0.2, 17, strategy='count_stratified', sample_counts=counts
    )
    train, validation = first

    assert first == second
    assert len(train) == 80
    assert len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(100))
    assert min(validation) < 10
    assert max(validation) > 89
