#!/usr/bin/python
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

def create_logger() -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param save_dir: The directory in which to save the logs.
    :return: The logger.
    """

    logger = logging.getLogger('final.log')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create file handler which logs even debug messages
    fh = logging.FileHandler('output.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def delta_target(train, test):

    X = np.array(train['dG_rxn'].values.tolist()).reshape(-1, 1)
    y = np.array(train['DG_TS_tunn'].values.tolist()).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    for data in [train, test]:
        data['DG_TS_tunn_linear'] = model.predict(data[['dG_rxn']])
        data['ddG_TS_tunn'] = data['DG_TS_tunn'] - data['DG_TS_tunn_linear']

    return train, test
