import numpy as np
import matplotlib.pyplot as plt


def promptM() -> int:
    print(f">>    Insert m+1 - the number of points")
    inpt = input()
    try:
        m = int(inpt) - 1
    except ValueError:
        print(f">>    Error: {inpt} is not an int value")
        return promptM()
    if m < 9:
        print(f">>    Error: {m+1} < 10 and m+1 must be more or equal to 10")
        return promptM()
    return m


def promptA() -> float:
    print(f">>    Insert a - the starting point")
    inpt = input()
    try:
        a = float(inpt)
    except ValueError:
        print(f">>    Error: {inpt} is not a float value")
        return promptA()
    return a


def promptH() -> float:
    print(f">>    Insert h - the step value")
    inpt = input()
    try:
        h = float(inpt)
    except ValueError:
        print(f">>    Error: {inpt} is not a float value")
        return promptH()
    if h <= 0:
        print(f">>    Error: {inpt} must be more than 0")
        return promptH()
    return h



