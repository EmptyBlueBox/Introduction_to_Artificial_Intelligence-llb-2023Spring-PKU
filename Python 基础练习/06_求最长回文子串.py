a = input()
l = len(a)
dp = [[0 for i in range(l)]for i in range(l)]
for i in range(l):
    dp[i][i] = 1
ans = 0
for j in range(0, l):
    for i in range(0, j):
        if (a[i] == a[j]):
            if (j-i == 1 or j-i == 2):
                dp[i][j] = 1
            else:
                dp[i][j] = dp[i+1][j-1]
            if (dp[i][j]):
                ans = max(ans, j-i+1)
print(ans)
