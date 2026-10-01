"""Comparable public-only NQueens candidate commands for a new qualification.

The negative changes one diagonal constraint in the same full public program.
Original task facts and the original grader remain unchanged.
"""
from .native_prolog_actor import nqueens_command
VERSION='public-nqueens-diagonal-candidates-v1'

def candidates(public):
    positive=nqueens_command(public)
    marker='abs(Q-R) #\\= D'
    if positive.count(marker)!=1:raise ValueError('qualified public diagonal program')
    negative=positive.replace(marker,'abs(Q-R) #\\=0')
    return [positive,negative]
