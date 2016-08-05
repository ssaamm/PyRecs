import itertools as it
import numpy as np
import pandas as pd
import pyrecs
import pytest
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

list_data = [[10,     3.4, np.nan, None],
             [10,     0,   10,     5],
             [np.nan, 1.4, 10,     3],
             [np.nan, 8,   2,      5]]
data = np.array(list_data, dtype=np.float64)

X, y = ([(0, 0), (0, 1),
         (1, 0), (1, 1), (1, 2), (1, 3),
         (2, 1), (2, 2), (2, 3),
         (3, 1), (3, 2), (3, 3)],
        [10, 3.4,
         10, 0, 10, 5,
         1.4, 10, 3,
         8, 2, 5])

@pytest.mark.parametrize('data', [data, list_data])
def test_cf_predictions(data):
    X, y = pyrecs.collab.matrix_to_dataset(data)
    cf = pyrecs.collab.CollaborativeFiltering()

    cf.fit(X, y)
    positive_preds = cf.predict([(0, 2), (2, 0)])
    negative_preds = cf.predict([(3, 0)])

    for pos, neg in it.product(positive_preds, negative_preds):
        assert pos > neg


def test_sparse_data():
    X, y = pyrecs.collab.matrix_to_dataset([[10, None],
                                            [None, 10]])
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(X, y)
    cf.predict([(0, 1), (1, 0)])


def test_double_fit():
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(X, y)

    with pytest.raises(RuntimeError):
        cf.fit(X, y)


def test_fit_without_target():
    cf = pyrecs.collab.CollaborativeFiltering()
    with pytest.raises(RuntimeError):
        cf.fit(X)


def test_predict_before_fit():
    cf = pyrecs.collab.CollaborativeFiltering()
    with pytest.raises(RuntimeError):
        cf.predict([(0, 2)])


def test_predict_non_iterable():
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(X, y)

    with pytest.raises(TypeError):
        cf.predict((0, 2))


def test_predict_out_of_range():
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(X, y)

    with pytest.raises(IndexError):
        cf.predict([(10, 10)])


@pytest.mark.parametrize('data', [data, list_data])
def test_rating_matrix_to_dataset(data):
    X, y = pyrecs.collab.matrix_to_dataset(data)

    expected_tuples = [(0, 0, 10), (0, 1, 3.4),
                       (1, 0, 10), (1, 1, 0), (1, 2, 10), (1, 3, 5),
                       (2, 1, 1.4), (2, 2, 10), (2, 3, 3),
                       (3, 1, 8), (3, 2, 2), (3, 3, 5)]

    X_expect = [(r, c) for r, c, _ in expected_tuples]
    y_expect = [v for _, _, v in expected_tuples]

    assert X == X_expect
    assert y == y_expect


def test_dataset_to_matrix():
    matrix = pyrecs.collab.dataset_to_matrix(X, y)

    for r_test, r_expect in zip(matrix, data):
        print(r_test, r_expect)
        for v_test, v_expect in zip(r_test, r_expect):
            assert ((np.isnan(v_test) and np.isnan(v_expect))
                    or v_test == v_expect)


def test_index_mapping():
    prefs = os.path.join(BASE_DIR, 'data/restaurant_preferences.csv')
    df = pd.read_csv(prefs, index_col=0)

    print(df)

    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(df)

    predictions = cf.predict([('Sam', 'Tacodeli')])
    assert predictions[0] > 2.5


def test_index_nonunique_index_vals():
    prefs = os.path.join(BASE_DIR, 'data/nonunique_rows.csv')
    df = pd.read_csv(prefs, index_col=0)

    cf = pyrecs.collab.CollaborativeFiltering()
    with pytest.raises(ValueError):
        cf.fit(df)
