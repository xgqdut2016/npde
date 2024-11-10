#排大运 阳男阴女逆排
from birthSihua import *
from anXing import *

# def liuNian(ind,step):

#待修改。能否按大运输出。
def daYun(ind,step):
    if Wuxingju == '水二局':
        startNum = 2
    elif Wuxingju == '木三局':
        startNum = 3
    elif Wuxingju == '金四局':
        startNum = 4
    elif Wuxingju == '土五局':
        startNum = 5
    elif Wuxingju == '火六局':
        startNum = 6

    A = (a.lunarYear+6) % 10
    if (A % 2 == 0 and a.gender == '男'
        or A % 2 == 1 and a.gender == '女'):
            print(startNum + ind%12 * 10, '-', startNum + ind %12* 10 + 9, TianganList[(ind - step) % 12],
              dizhilist[(-ind + step) % 12])
            for j in range(1, 13):
                if gongname[(-ind + j - 1) % 12] =='命宫':
                    print("第", j, "大运：" + TianganList[(-step + j - 1) % 12] + gongname[(-ind + j - 1) % 12])
                    birthSihua(starlist[(-ind + step) % 12], (-ind + j - 1) % 12, TianganList[(-step + j - 1) % 12])
                    if gongname[(-ind + j - 1) % 12] == "命宫":
                        print(a.lunarYear+startNum+ind %12* 10-1,'-',a.lunarYear+startNum+ind%12 * 10+9-1)
                        for j2 in range(0,10):
                            print(a.lunarYear+startNum+ind %12* 10-1+j2)
                            # 天干数b,地支数d
                            b=(a.lunarYear+startNum+ind %12* 10-1+j2+6)%10
                            d=(a.lunarYear+startNum+ind %12* 10+j2+3)%12
                            for j3 in range(0,12):
                                birthSihua(starlist[( + j3) % 12], (d+j3)%12, TianganList2[b])
    else:
        print(startNum + (12 - ind)  * 10, '-', startNum + (12 - ind)  * 10 + 9, TianganList[(ind - step) % 12],
              dizhilist[(-ind + step) % 12])
        for j in range(1, 13):
            if gongname[(-ind - j + 1) % 12] == '命宫':
                print("第", j, "大运：" + TianganList[(-step - j + 1) % 12] + ',' + gongname[(-ind - j + 1) % 12])
                birthSihua(starlist[(-ind + step) % 12], (-ind - j + 1) % 12, TianganList[(-step - j + 1) % 12])
                if gongname[(-ind - j + 1) % 12] == '命宫':
                    print(a.lunarYear+startNum+(12-ind) * 10-1,'-',a.lunarYear+startNum+(12-ind) * 10+9-1)
                    for j2 in range(0, 10):
                        print(a.lunarYear + startNum + (12-ind) % 12 * 10 - 1 + j2)
                        # 天干数b,地支数d
                        b = (a.lunarYear + startNum + (12-ind) % 12 * 10 - 1 + j2 + 6) % 10
                        d = (a.lunarYear + startNum + (12-ind )% 12 * 10 + j2 + 3) % 12
                        for j3 in range(0, 12):
                            birthSihua(starlist[( - j3) % 12], (d-j3)%12, TianganList2[b])
#大运、宫位、四化
# def daYun2(ind,step):
#     if Wuxingju == '水二局':
#         startNum = 2
#     elif Wuxingju == '木三局':
#         startNum = 3
#     elif Wuxingju == '金四局':
#         startNum = 4
#     elif Wuxingju == '土五局':
#         startNum = 5
#     elif Wuxingju == '火六局':
#         startNum = 6
#
#     A = (a.lunarYear+6) % 10
#     if (A % 2 == 0 and a.gender == '男'
#         or A%2 == 1 and a.gender == '女'):
#             print(startNum+ind*10,'-',startNum+ind*10+9,TianganList[(ind -step) % 12],dizhilist[(-ind + step) % 12])
#             for j in range(1, 13):
#                 print("第", j, "大运：" +TianganList[( -step+j-1) % 12]+ gongname[(-ind + j - 1) % 12])
#                 birthSihua(starlist[(-ind + step) % 12], (-ind+j-1)%12,TianganList[( -step+j-1) % 12])
#     else :
#             print(startNum+(12-ind)*10,'-',startNum+(12-ind)*10+9,TianganList[(ind -step) % 12],dizhilist[(-ind + step) % 12])
#             for j in range(1,13):
#                 print("第",j,"大运："+TianganList[( -step-j+1) % 12]+','+gongname[(-ind-j+1)%12])
#                 birthSihua(starlist[(-ind + step) % 12], (-ind-j+1)%12,TianganList[( -step-j+1) % 12])