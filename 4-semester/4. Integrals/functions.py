
def promptPair() -> list[float]:

    print(f">>    Insert a, b")
    inpt = [x for x in input().split()]
    if len(inpt) < 2:
        print(f">>    Error: Not enough values")
        return promptPair()
    try:
        a = float(inpt[0])
    except ValueError:
        print(f">>    Error: {inpt[0]} is not a float value")
        return promptPair()
    try:
        b = float(inpt[1])
    except ValueError:
        print(f">>    Error: {inpt[1]} is not a float value")
        return promptPair()
    if b <= a:
        print(f">>    Error: {a} >= {b} but a must be less than b")
        return promptPair()
    return [a, b]

