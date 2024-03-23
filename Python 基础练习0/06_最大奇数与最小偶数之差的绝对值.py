while True:
    try:
        a = list(map(int, input().split(' ')))
        M_1 = -1
        m_0 = 102
        for i in a:
            if (i % 2):
                M_1 = max(M_1, i)
            else:
                m_0 = min(m_0, i)
        print(abs(M_1-m_0))
    except EOFError:
        break
