# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:48:34 2020

@author: marquart
"""
import logging

import numpy as np


def input_check(times=None, tokens=None):
    """
    Check provided inputs and throw errors if needed.

    Parameters
    ----------
    times : TYPE, optional
        Year or time variable used in int or dict format. The default is None.
    tokens : TYPE, optional
        appropriate list or array. The default is None.

    Returns
    -------
    times and tokens, possibly cast to basic python types

    """
    if times is not None:
        if not isinstance(times, (np.ndarray, list, dict, int, np.integer)):
            logging.error(
                "Parameter times must be int, list, array or interval dict ('start':int,'end':int). Supplied: {}".format(
                    type(times)))
            raise AttributeError(
                "Parameter times must be int, list, array or interval dict ('start':int,'end':int). Supplied: {}".format(
                    type(times)))
        else:
            # We use lists instead of arrays here
            if isinstance(times, np.ndarray):
                times = times.tolist()
            # If a list, we do need Python builtins instead of np data types due to database requirements
            if isinstance(times, list):
                if not isinstance(times[0], int):
                    try:
                        times = [int(x) for x in times]
                    except:
                        msg = "Parameter times includes elements that could not be cast to integer"
                        logging.error(msg)
                        raise AttributeError(msg)
            if isinstance(times, dict):
                for key in times:
                    try:
                        times[key] = int(times[key])
                    except:
                        msg = "Parameter times includes elements that could not be cast to integer"
                        logging.error(msg)
                        raise AttributeError(msg)
    if token is not None:
        if not isinstance(tokens, (np.ndarray, list, int, np.integer, np.str_, np.string_)):
            logging.error("Token parameter should be string, int or list. Supplied: {}".format(type(tokens)))
            raise AttributeError("Token parameter should be string, int or list. Supplied: {}".format(type(tokens)))
        else:
            # We use lists instead of arrays here
            if isinstance(tokens, np.ndarray):
                tokens = times.tolist()
            # If a list, we do need Python builtins instead of np data types due to database requirements
            if isinstance(tokens, list):
                for i, token in enumerate(tokens):
                    if isinstance(token, (np.integer)):
                        tokens[i] = int(token)
                    elif isinstance(token, (np.str_, np.string_)):
                        tokens[i] = str(token)
                    elif not isinstance(token, (str, int)):
                        msg = "Parameter times includes elements that could not be cast to integer"
                        logging.error(msg)
                        raise AttributeError(msg)
            if isinstance(times, dict):
                for key in times:
                    try:
                        times[key] = int(times[key])
                    except:
                        msg = "Parameter times includes elements that could not be cast to integer"
                        logging.error(msg)
                        raise AttributeError(msg)

    if tokens is not None and times is not None:
        return times, tokens
    elif tokens is not None:
        return tokens
    else:
        return times
