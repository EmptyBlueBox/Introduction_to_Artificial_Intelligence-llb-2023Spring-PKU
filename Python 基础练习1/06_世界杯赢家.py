n = int(input())
x = set()
y = set()
for i in range(n):
    a, b = map(int, input().split())
    x.add(a), y.add(b)
ans = list(x-y)
ans.sort()  # 只能给list排序
print(", ".join(map(str, ans)))  # join只能应用于字符串
