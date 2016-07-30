import itertools as it
import numpy as np
import pyrecs
import pytest

list_data = [[10,     3.4, np.nan, None],
             [10,     0,   10,     5],
             [np.nan, 1.4, 10,     3],
             [np.nan, 8,   2,      5]]
data = np.array(list_data, dtype=np.float64)


@pytest.mark.parametrize('data', [data, list_data])
def test_cf_predictions(data):
    cf = pyrecs.collab.CollaborativeFiltering()

    cf.fit(data)
    positive_preds = cf.predict([(0, 2), (2, 0)])
    negative_preds = cf.predict([(3, 0)])

    for pos, neg in it.product(positive_preds, negative_preds):
        assert pos > neg


def test_double_fit():
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(data)

    with pytest.raises(RuntimeError):
        cf.fit(data)


def test_predict_before_fit():
    cf = pyrecs.collab.CollaborativeFiltering()
    with pytest.raises(RuntimeError):
        cf.predict([(0, 2)])


def test_predict_non_iterable():
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(data)

    with pytest.raises(TypeError):
        cf.predict((0, 2))


def test_predict_out_of_range():
    cf = pyrecs.collab.CollaborativeFiltering()
    cf.fit(data)

    with pytest.raises(IndexError):
        cf.predict([(10, 10)])


def test_not_2d():
    cf = pyrecs.collab.CollaborativeFiltering()

    with pytest.raises(ValueError):
        cf.fit([1, 2, 3])
        print(cf.predict([(0, 1)]))
