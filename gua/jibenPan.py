# import datetime
# import cnlunar

# from daYun import *
# from anXing import *
from zaYao import *
from gongV11 import *
from gongV22 import *
from birthSihua import *
birthTiangan = a.year8Char[0]
#------------按地支加入杂曜
def predict(ind, step):
    print(gongname[ind] + "如下：--------------------------------------------------------------------------------------")
    if(ind == 0):
        GongChu1st(starlist[(ind + step) % 12],dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12],ind,birthTiangan)
    elif(ind == 1):
        GongChu2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 2):
        GongChu3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 3):
        GongChu4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 4):
        GongChu5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 5):
        GongChu6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 6):
        GongChu7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 7):
        GongChu8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 8):
        GongChu9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 9):
        GongChu10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
        GongXia10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 10):
        GongChu11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind - step) % 12])
        GongXia11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
    elif(ind == 11):
        GongChu12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind - step) % 12])
        GongXia12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        ZaYao((ind + step) % 12,ind)
        birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)



# def predict2(ind, step):
#     print(gongname[ind] + "如下：--------------------------------------------------------------------------------------")
#     if(ind == 0):
#         # GongChu1st(starlist[(ind + step) % 12],dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12],ind,birthTiangan)
#     elif(ind == 1):
#         # GongChu2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 2):
#         # GongChu3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 3):
#         # GongChu4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 4):
#         # GongChu5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 5):
#         # GongChu6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 6):
#         # GongChu7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 7):
#         # GongChu8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 8):
#         # GongChu9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 9):
#         # GongChu10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind -step) % 12])
#         # GongXia10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 10):
#         # GongChu11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind - step) % 12])
#         # GongXia11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
#     elif(ind == 11):
#         # GongChu12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],TianganList[(-ind - step) % 12])
#         # GongXia12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#         # ZaYao((ind + step) % 12,ind)
#         birthSihua(starlist[(ind + step) % 12], ind,birthTiangan)
# if MingDizhi =='辰':
#     step = 0
#---------- DizhiNum是从辰顺时针转,  12-DizhiNum是从辰逆时针转
# def result(birthDate,gender):
for ind in range(12):
    predict(ind, 12-DizhiNum)
    # daYun(12-ind,12-DizhiNum)

# for ind in range(12):
#     daYun2(12-ind,12-DizhiNum)