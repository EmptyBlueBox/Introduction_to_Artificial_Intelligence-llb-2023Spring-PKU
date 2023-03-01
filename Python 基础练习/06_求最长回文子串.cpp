#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
using namespace std;
const int maxn = 1e6 + 5;
char s[maxn * 2], str[maxn * 2];
int Len[maxn * 2], len;
void getstr()
{ // 重定义字符串
    int k = 0;
    str[k++] = '@'; // 开头加个特殊字符防止越界
    for (int i = 0; i < len; i++)
    {
        str[k++] = '#';
        str[k++] = s[i];
    }
    str[k++] = '#';
    len = k;
    str[k] = 0; // 字符串尾设置为0，防止越界
}
int manacher()
{
    int mx = 0, id; // mx为最右边，id为中心点
    int maxx = 0;
    for (int i = 1; i < len; i++)
    {
        if (mx > i)
            Len[i] = min(mx - i, Len[2 * id - i]); // 判断当前点超没超过mx
        else
            Len[i] = 1; // 超过了就让他等于1，之后再进行查找
        while (str[i + Len[i]] == str[i - Len[i]])
            Len[i]++; // 判断当前点是不是最长回文子串，不断的向右扩展
        if (Len[i] + i > mx)
        { // 更新mx
            mx = Len[i] + i;
            id = i;                   // 更新中间点
            maxx = max(maxx, Len[i]); // 最长回文字串长度
        }
    }
    return (maxx - 1);
}
int main()
{
    scanf("%s", s);
    len = strlen(s);
    getstr();
    printf("%d\n", manacher());
    return 0;
}