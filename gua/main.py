from gongV1 import *
from gongV2 import *

star1st = "贪狼"
star2nd = "太阴"
star3rd = "紫薇天府"
star4th = "天机"
star5th = "破军"
star6th = "太阳"
star7th = "武曲"
star8th = "天同"
star9th = "七杀"
star10th = "天梁"
star11th = "廉贞天相"
star12th = "巨门"
starlist =[star1st, star2nd, star3rd, star4th, star5th, star6th, star7th, star8th, star9th, star10th, star11th, star12th]

dizhi1st = "辰"
dizhi2nd = "卯"
dizhi3rd = "寅"
dizhi4th = "丑"
dizhi5th = "子"
dizhi6th = "亥"
dizhi7th = "戌"
dizhi8th = "酉"
dizhi9th = "申"
dizhi10th = "未"
dizhi11th = "午"
dizhi12th = "巳"
dizhilist =[dizhi1st, dizhi2nd, dizhi3rd, dizhi4th, dizhi5th, dizhi6th, dizhi7th, dizhi8th, dizhi9th, dizhi10th, dizhi11th, dizhi12th]
gongname = ["命宫", "兄弟宫", "夫妻宫", "子女宫", "财帛宫", "疾厄宫", "迁移宫", "交友宫", "官禄宫", "田宅宫", "福德宫", "父母宫"]
def predict(ind, step):
    print(gongname[ind] + "：", end = " ")
    if(ind == 0):
        GongChu1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 1):
        GongChu2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 2):
        GongChu3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 3):
        GongChu4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 4):
        GongChu5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 5):
        GongChu6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 6):
        GongChu7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 7):
        GongChu8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 8):
        GongChu9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 9):
        GongChu10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 10):
        GongChu11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 11):
        GongChu12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
        GongXia12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#step = 6
step = 0
for ind in range(12):
    predict(ind, step)