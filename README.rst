======
PyRecs
======

.. image:: https://api.travis-ci.org/ssaamm/PyRecs.svg?branch=master&service=github
    :target: https://travis-ci.org/ssaamm/PyRecs?branch=master
.. image:: https://coveralls.io/repos/github/ssaamm/PyRecs/badge.svg?branch=master
    :target: https://coveralls.io/github/ssaamm/PyRecs?branch=master


A **Py**\ thon **Rec**\ ommender **S**\ ystems library

Installation
------------

I have only tested on Python 3.5.1

.. code:: bash

    pip3 install pyrecs

Usage
-----

Predicting ratings based on training data:

.. code:: python

    >>> import numpy as np
    >>> import pyrecs
    >>> from sklearn.cross_validation import train_test_split
    >>>
    >>> data = [[10,     3.4, np.nan, None],
    ...         [10,     0,   10,     5],
    ...         [np.nan, 1.4, 10,     3],
    ...         [np.nan, 8,   2,      5]]
    >>>
    >>> X, y = pyrecs.collab.matrix_to_dataset(data)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> cf = pyrecs.collab.CollaborativeFiltering()
    >>> cf.fit(X_train, y_train)
    >>>
    >>> cf.predict(X_test)
    array([ 0.25,  3.4 ,  9.75])
    >>> y_test
    [1.4, 10, 8]


Tests
-----

.. code:: bash

    py.test tests
