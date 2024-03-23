import re
n, c = input().split(" ")
n = int(n)
c = int(float(c)*100)
info = []
for i in range(n):
    tmp = input()
    info.append(list(re.split(r"[ ]+", tmp)))
    info[i][0] = int(float(info[i][0])*100)
    info[i][1] = float(info[i][1])
dp = [0]*(c+1)
for i in range(n):
    for j in range(c, info[i][0], -1):
        dp[j] = max(dp[j], dp[j-info[i][0]]+info[i][1])
print("%.5f" % dp[c])
