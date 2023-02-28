dp = [[0 for i in range(11)]for j in range(101)]
for i in range(1, 101):
    dp[i][1] = i
for i in range(1, 11):
    dp[1][i] = 1
for i in range(2, 101):
    for j in range(2, 11):
        dp[i][j] = 0x3f3f3f3f
for i in range(2, 101):
    for j in range(2, 11):
        for k in range(1, i+1):
            dp[i][j] = min(dp[i][j], 1+max(dp[i-k][j], dp[k-1][j-1]))

while True:
    try:
        a, b = map(int, input().split(" "))
        print(dp[a][b])
    except EOFError:
        break
