from gongV1 import *
from gongV2 import *
allStar = ['紫薇','天机','太阳','武曲','天同','廉贞','天府','太阴','贪狼','巨门','天相','天梁','七杀','破军']
star1st = allStar[0]
star2nd = allStar[1]
star3rd = allStar[2]
star4th = allStar[3]
star5th = allStar[4]
star6th = allStar[5] + allStar[0]
star7th = allStar[6]
star8th = allStar[7]
star9th = allStar[8]
star10th = allStar[9]
star11th = allStar[10]
star12th = allStar[11]
starlist =[star1st, star2nd, star3rd, star4th, star5th, star6th, star7th, star8th, star9th, star10th, star11th, star12th]

#辅佐八曜
fuyao = ['天魁','天铖','左辅','右弼']
zuoyao = ['文昌','文曲','禄存','天马']

shayao = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']#四大煞曜+空劫=煞曜

kongyao = ['天空','空劫','截空','旬空']#空曜

xinyao = ['天刑','擎羊']#刑曜

jiyao = ['化忌','陀罗']#忌曜

taohua = ['咸池', '红鸾', '天喜','沐浴',  '天姚' ,'大耗']#桃花曜

huayao = ['化禄', '化科', '化权','化忌']#化曜

tian_good = ['天德','天寿','天贵','天官','天福','天巫','天才']#天开头吉曜
tian_bad = ['天伤', '天哭','天虚','天月']#天开头煞曜


caiyi = ['三台', '八座', '龙池','凤阁','天才','封诰','华盖','台辅']#才艺官运曜

za_good = ['恩光','解神','月德','长生','冠带','临官','帝旺']#杂吉曜
za_bad = ['蜚廉','孤辰', '寡宿', '破碎','阴煞']#杂坏曜


smallStar1st = [fuyao[0], shayao[0], tian_good[0], tian_bad[1], taohua[1], caiyi[1], za_good[1], za_bad[1]]
smallStar2nd = [fuyao[1], shayao[1], tian_good[1], tian_bad[1], taohua[1], caiyi[1], za_good[1], za_bad[1]]
smallStar3rd = [fuyao[1], shayao[1], tian_good[1], tian_bad[2], taohua[5], caiyi[1], za_good[1], za_bad[1]]
smallStar4th = [fuyao[3], shayao[1], tian_good[1], tian_bad[1], taohua[1], caiyi[3], za_good[1], za_bad[1]]
smallStar5th = [fuyao[1], shayao[3], tian_good[1], tian_bad[1], taohua[1], caiyi[1], za_good[3], za_bad[1]]
smallStar6th = [fuyao[2], shayao[1], tian_good[2], tian_bad[1], taohua[1], caiyi[1], za_good[1], za_bad[1]]
smallStar7th = [fuyao[1], shayao[0], tian_good[1], tian_bad[1], taohua[1], caiyi[0], za_good[1], za_bad[1]]
smallStar8th = [fuyao[1], shayao[1], tian_good[1], tian_bad[1], taohua[1], caiyi[1], za_good[4], za_bad[4]]
smallStar9th = [fuyao[0], shayao[1], tian_good[1], tian_bad[1], taohua[1], caiyi[0], za_good[1], za_bad[1]]
smallStar10th = [fuyao[1], shayao[1], tian_good[3], tian_bad[1], taohua[1], caiyi[1], za_good[1], za_bad[0]]
smallStar11th = [fuyao[1], shayao[1], tian_good[1], tian_bad[3], taohua[1], caiyi[5], za_good[1], za_bad[1]]
smallStar12th = [fuyao[0], shayao[1], tian_good[1], tian_bad[1], taohua[0], caiyi[1], za_good[1], za_bad[1]]
smallStarList = [smallStar1st, smallStar2nd, smallStar3rd, smallStar4th, smallStar5th, smallStar6th, smallStar7th, smallStar8th, smallStar9th, smallStar10th, smallStar11th, smallStar12th]
dizhitmp = ['亥', '戌', '酉', '申', '未', '午', '巳', '辰', '卯', '寅', '丑', '子']

pian = 6#默认命宫为亥，如果命宫在巳，偏移量为6
dizhilist = []
for i in range(12):
    dizhilist.append(dizhitmp[(i + pian)%12])


gongname = ["命宫", "兄弟宫", "夫妻宫", "子女宫", "财帛宫", "疾厄宫", "迁移宫", "交友宫", "官禄宫", "田宅宫", "福德宫", "父母宫"]
def predict(ind, step):
    print(gongname[ind] + "：", end = " ")
    if(ind == 0):
        GongChu1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia1st(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 1):
        GongChu2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia2nd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 2):
        GongChu3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia3rd(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 3):
        GongChu4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia4th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 4):
        GongChu5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia5th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 5):
        GongChu6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia6th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 6):
        GongChu7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia7th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 7):
        GongChu8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia8th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 8):
        GongChu9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia9th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 9):
        GongChu10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia10th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 10):
        GongChu11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia11th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
    elif(ind == 11):
        GongChu12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12],smallStarList[(ind + step) % 12])
        GongXia12th(starlist[(ind + step) % 12], dizhilist[(ind + step) % 12])
#step = 6
step = 0
for ind in range(12):
    predict(ind, step)
