n = int(input())
a = list(map(int, input().split(' ')))
a.sort()
if n % 2:
    print(a[n//2])
else:
    print((a[n//2]+a[n//2-1])/2 if (a[n//2]+a[n//2-1]) %
          2 else int((a[n//2]+a[n//2-1])/2))
