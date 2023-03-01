# 输入一个列表，返回[最大连续子序列和, 起始位置, 终止位置]
def max_continuous_sequence(a):
    dp = [[0, 0] for i in range(len(a))]
    max = [-2147483648, 0, 0]
    for i in range(len(a)):
        if dp[i-1][0] > 0:
            dp[i][0] = dp[i-1][0]+a[i]
            dp[i][1] = dp[i-1][1]
        else:
            dp[i][0] = a[i]
            dp[i][1] = i
        if (dp[i][0] > max[0]):
            max[0] = dp[i][0]
            max[1] = dp[i][1]
            max[2] = i
    return max


n = int(input())
a = []
for i in range(n):
    a.append(list(map(int, input().split(" "))))
ans = [-2147483648, 0, 0, 0, 0]  # [最大子矩阵和, 行起始位置, 行结束位置, 列起始位置, 列结束位置]
for i in range(n):
    b = [0]*n
    for j in range(i, n):
        b = list(map(lambda x, y: x+y, b, a[j]))
        tmp = max_continuous_sequence(b)
        if (tmp[0] > ans[0]):
            ans[0] = tmp[0]
            ans[1] = i
            ans[2] = j
            ans[3] = tmp[1]
            ans[4] = tmp[2]
for i in range(ans[1], ans[2]+1):
    for j in range(ans[3], ans[4]):
        print(a[i][j], end=' ')
    print(a[i][ans[4]])
print(ans[0])
