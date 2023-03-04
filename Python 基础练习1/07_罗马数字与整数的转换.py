a = input()
t = [[1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'], [100, 'C'], [90, 'XC'], [
    50, 'L'], [40, 'XL'], [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I']]
if a[0] >= '0' and a[0] <= '9':
    num = int(a)
    ans = ''
    for x in t:
        while num >= x[0]:
            num -= x[0]
            ans += x[1]
    print(ans)
else:
    roma = a
    ans = 0
    for x in t:
        while roma.find(x[1]) == 0:
            roma = roma[len(x[1]): len(roma)]  # 不能用lstrip，可能会删除多个一样的罗马数字
            ans += x[0]
    print(ans)
