from funcs import f, df, epsilon, get_N, PrepareOutput, get_x, get_x0_n, partition


SegmentList = partition(get_N())

for i in range(len(SegmentList)):
    print(f"{i + 1}: {SegmentList[i]}")


def TestInput(inp):
    try:
        tmp = int(inp)
    except ValueError:
        return True
    if (str(int(inp)) == inp) and (int(inp) - 1 >= 0) and (int(inp) - 1 < len(SegmentList)):
        return False
    else:
        return True


i = input()
while TestInput(i):
    print(f"Wrong Input: {i} is not an available index")
    i = input()
else:
    i = int(i)
    Segment = SegmentList[i - 1]

print(f"Chosen segment: {Segment}")


def Bisection(segment):
    left = segment[0]
    right = segment[1]
    cnt = 0
    cen = left + (right - left) / 2
    if f(cen) == 0:
        print(cen)
        return
    else:
        while (right - left) / 2 >= epsilon():
            cnt += 1
            if f(cen) * f(right) < 0:
                left = cen
            else:
                right = cen
            cen = left + (right - left) / 2
            if f(cen) == 0:
                break
    PrepareOutput("Метод Бисекции", (segment[1] + segment[0]) / 2, cnt, cen, (right - left) / 2, abs(f(cen)))
    return


def Newton(segment):
    x0 = get_x0_n(segment)
    xprev = x0
    x1 = xprev - f(xprev) / df(xprev)
    xnext = x1
    cnt = 0
    while abs(xnext - xprev) >= epsilon():
        xnext, xprev = xnext - f(xnext) / df(xnext), xnext
        cnt += 1

    PrepareOutput("Метод Ньютона", x0, cnt, xnext, abs(xnext - xprev), abs(f(xnext)))

    xprev = x0
    x1 = xprev - f(xprev) / df(xprev)
    xnext = x1
    cnt = 0
    while abs(xnext - xprev) >= epsilon():
        xnext, xprev = xnext - f(xnext) / df(x0), xnext
        cnt += 1

    PrepareOutput("Модифицированный Метод Ньютона", x0, cnt, xnext, abs(xnext - xprev), abs(f(xnext)))


def Secants(segment):
    a = segment[0]
    b = segment[1]
    x0 = get_x(segment)
    # x0 = a
    xp = x0
    if f(a) * f(x0) >= 0:
        x1 = get_x([x0, b])
    else:
        x1 = get_x([a, x0])
    xc = x1
    cnt = 0
    xn = xc - (f(xc) / (f(xc) - f(xp))) * (xc - xp)
    while abs(xn - xc) >= epsilon():
        xn, xc, xp = xn - (f(xn) / (f(xn) - f(xc))) * (xn - xc), xn, xc
        cnt += 1
    PrepareOutput("Метод Секущих", [x0, x1], cnt, xn, abs(xn - xc), abs(f(xn)))


Bisection(Segment)
Newton(Segment)
Secants(Segment)
