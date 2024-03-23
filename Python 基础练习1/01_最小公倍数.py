def gcd(a, b):
    return gcd(b, a % b) if b else a


a, b = map(int, input().split(","))
print(int(a*b/gcd(a, b)))
