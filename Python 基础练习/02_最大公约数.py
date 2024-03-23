def gcd(a, b):
    return gcd(b, a % b) if b else a


n = int(input())
for i in range(n):
    a, b = map(int, input().split(" "))
    print(gcd(a, b))
