n, k = map(int, input().split(' '))
has = n
need = 200
ok = True
for i in range(1, 21):
    if (has >= need):
        print(i)
        ok = False
        break
    has += n
    need *= 1+k/100
if (ok):
    print('Impossible')
