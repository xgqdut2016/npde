from gongV1 import *
from gongV2 import *

star1st = "太阳"
star2nd = "武曲七杀"
star3rd = "天同天梁"
star4th = "天相"
star5th = "巨门"
star6th = "廉贞贪狼"
star7th = "太阴"
star8th = "天府"
star9th = "天同天梁"
star10th = "紫微破军"
star11th = "天机"
star12th = "廉贞贪狼"
starlist =[star1st, star2nd, star3rd, star4th, star5th, star6th, star7th, star8th, star9th, star10th, star11th, star12th]

#-----------------------------------------------
fuyao1st = ['天魁','天钺','左辅','右弼']
zuoyao1st = ['文昌','文曲','禄存','天马']
shayao1st = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao1st = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao2nd = ['天魁','天钺','左辅','右弼']
zuoyao2nd = ['文昌','文曲','禄存','天马']
shayao2nd = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao2nd = ['化禄', '化科', '化权','化忌']#化曜

#-----------------------------------------------
fuyao3rd = ['天魁','天钺','左辅','右弼']
zuoyao3rd = ['文昌','文曲','禄存','天马']
shayao3rd = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao3rd = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao4th = ['天魁','天钺','左辅','右弼']
zuoyao4th = ['文昌','文曲','禄存','天马']
shayao4th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao4th = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao5th = ['天魁','天钺','左辅','右弼']
zuoyao5th = ['文昌','文曲','禄存','天马']
shayao5th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao5th = ['化禄', '化科', '化权','化忌']#化曜

#-----------------------------------------------
fuyao6th = ['天魁','天钺','左辅','右弼']
zuoyao6th = ['文昌','文曲','禄存','天马']
shayao6th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao6th = ['化禄', '化科', '化权','化忌']#化曜



#-----------------------------------------------
fuyao7th = ['天魁','天钺','左辅','右弼']
zuoyao7th = ['文昌','文曲','禄存','天马']
shayao7th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao7th = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao8th = ['天魁','天钺','左辅','右弼']
zuoyao8th = ['文昌','文曲','禄存','天马']
shayao8th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao8th = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao9th = ['天魁','天钺','左辅','右弼']
zuoyao9th = ['文昌','文曲','禄存','天马']
shayao9th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao9th = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao10th = ['天魁','天钺','左辅','右弼']
zuoyao10th = ['文昌','文曲','禄存','天马']
shayao10th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao10th = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao11th = ['天魁','天钺','左辅','右弼']
zuoyao11th = ['文昌','文曲','禄存','天马']
shayao11th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao11th = ['化禄', '化科', '化权','化忌']#化曜


#-----------------------------------------------
fuyao12th = ['天魁','天钺','左辅','右弼']
zuoyao12th = ['文昌','文曲','禄存','天马']
shayao12th = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']
huayao12th = ['化禄', '化科', '化权','化忌']#化曜



#----------------------------------------
fuyaoList = [fuyao1st, fuyao2nd, fuyao3rd, fuyao4th, fuyao5th, fuyao6th, fuyao7th, fuyao8th, fuyao9th, fuyao10th, fuyao11th, fuyao12th]
zuoyaoList = [zuoyao1st, zuoyao2nd, zuoyao3rd, zuoyao4th, zuoyao5th, zuoyao6th, zuoyao7th, zuoyao8th, zuoyao9th, zuoyao10th, zuoyao11th, zuoyao12th]
shayaoList = [shayao1st, shayao2nd, shayao3rd, shayao4th, shayao5th, shayao6th, shayao7th, shayao8th, shayao9th, shayao10th, shayao11th, shayao12th]

huayaoList = [huayao1st, huayao2nd, huayao3rd, huayao4th, huayao5th, huayao6th, huayao7th, huayao8th, huayao9th, huayao10th, huayao11th, huayao12th]

dizhitmp = ['亥', '戌', '酉', '申', '未', '午', '巳', '辰', '卯', '寅', '丑', '子']

pian = 8#默认命宫为亥，如果命宫在巳，偏移量为6
dizhilist = []
for i in range(12):
    dizhilist.append(dizhitmp[(i + pian)%12])


gongname = ["命宫", "兄弟宫", "夫妻宫", "子女宫", "财帛宫", "疾厄宫", "迁移宫", "交友宫", "官禄宫", "田宅宫", "福德宫", "父母宫"]
def predict(ind, step):
    print(gongname[ind] + "：", end = " ")
    if(ind == 0):
        GongChu1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 1):
        GongChu2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 2):
        GongChu3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 3):
        GongChu4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 4):
        GongChu5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 5):
        GongChu6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 6):
        GongChu7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 7):
        GongChu8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 8):
        GongChu9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 9):
        GongChu10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 10):
        GongChu11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
    elif(ind == 11):
        GongChu12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
        GongXia12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],
                   fuyaoList[(ind + step) % 12],zuoyaoList[(ind + step) % 12],
                   shayaoList[(ind + step) % 12], huayaoList[(ind + step) % 12])
#step = 6
step = 0
for ind in range(12):
    predict(ind, step)
