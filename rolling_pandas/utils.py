import pandas as pd

def pct_change(obj, periods=1):
	return obj.pct_change(periods, axis=0)


def diff(obj, periods=1):
	return obj.diff(periods, axis=0)


def shift(obj, periods=1):
	return obj.shift(periods, axis=0)