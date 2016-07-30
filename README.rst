======
PyRecs
======

A **Py**\ thon **Rec**\ ommender **S**\ ystems library

Installation
------------

I have only tested on Python 3.5.1

.. code:: bash

    pip3 install pyrecs

Usage
-----

Given a NUM_USERS x NUM_ITEMS matrix of ratings, predict the rating by user 0 of
item 2 and by user 2 of item 0:

.. code:: python

    >>> cf = pyrecs.collab.CollaborativeFiltering()
    >>> cf.fit([[10,     3.4, np.nan, None],
    ...         [10,     0,   10,     5],
    ...         [np.nan, 1.4, 10,     3],
    ...         [np.nan, 8,   2,      5]])
    >>> cf.predict([(0, 2), (2, 0)])
    array([ 10.68567893,   8.31302514])

Tests
-----

.. code:: bash

    py.test tests
