from anXing import *
from zaYaoDetail import *

#输出杂曜，此时都是顺时针计算

def ZaYao(dizhi, ind):
    print("杂曜：")
    # zuoyaoList=[]
#----------安时系诸星----------
    if (a.twohourNum + (12-dizhi))%12 == 6:
        # fuyaoList[dizhi] = '文昌'
        ZaYaoDetail('文昌')
    if a.twohourNum == (12-dizhi):
        # fuyaoList[dizhi] = '文曲'
        ZaYaoDetail('文曲')

    if (((a.year8Char[1] in ['寅','午','戌'] and ((12-dizhi) - a.twohourNum) in [-3,9] or
            a.year8Char[1] in ['申','子','辰'] and ((12-dizhi) - a.twohourNum) in [-2,10]) or
            a.year8Char[1] in ['巳', '酉', '丑'] and ((12-dizhi) - a.twohourNum) in [-1,11]) or
            a.year8Char[1] in ['亥','卯','未'] and ((12-dizhi) - a.twohourNum) in [-7,5]):
        ZaYaoDetail('火星')

    if (a.year8Char[1] in ['寅','午','戌'] and ((12-dizhi) - a.twohourNum) in [-1,11] or
            a.year8Char[1] not in ['寅','午','戌'] and ((12-dizhi) - a.twohourNum) in [-6,6]):
        ZaYaoDetail('铃星')

    if ((12-dizhi) - a.twohourNum)%12==7:
        ZaYaoDetail('地劫')
    if ((12-dizhi) + a.twohourNum)%12==7:
        ZaYaoDetail('地空')

    if ((12-dizhi) - a.twohourNum) in [2,-10]:
        ZaYaoDetail('台辅')
    if ((12-dizhi) - a.twohourNum) in [-2,10]:
        ZaYaoDetail('封诰')

#----------安月系诸星----------
    if ((12-dizhi) - a.lunarMonth) in [-1,11]:
        ZaYaoDetail('左辅')
    elif ((12-dizhi) + a.lunarMonth) in [7,19]:
        ZaYaoDetail('右弼')
    elif ((12-dizhi) - a.lunarMonth) in [-8,4]:
        ZaYaoDetail('天刑')
    elif ((12-dizhi) - a.lunarMonth) in [-4,8]:
        ZaYaoDetail('天姚')

    if (12-dizhi) in [1,4,7,10] :
        for i in range(-1,2):
            if a.lunarMonth == ((12-dizhi) + 1 + 4*i):
                ZaYaoDetail('天马')

    # if (12-dizhi) == 1 and a.lunarMonth in [3, 7, 11]:
    #     ZaYaoDetail('天马')
    # elif (12-dizhi) == 4 and a.lunarMonth in [1,5,9]:
    #     ZaYaoDetail('天马')
    # elif (12-dizhi) == 7 and a.lunarMonth in [4, 8, 12]:
    #     ZaYaoDetail('天马')
    # elif (12-dizhi) == 10 and a.lunarMonth in [2, 6, 10]:
    #     ZaYaoDetail('天马')

    if (12-dizhi)%2==0:
        if (a.lunarMonth%2==1 and ((12-dizhi)-a.lunarMonth) in [3,-9] or
                a.lunarMonth%2==0 and ((12-dizhi)-a.lunarMonth) in [2,-10]):
            ZaYaoDetail('解神')

    if ((12-dizhi) == 1 and a.lunarMonth in [1,5,9] or
        (12-dizhi) == 4 and a.lunarMonth in [3, 7, 11] or
        (12-dizhi) == 7 and a.lunarMonth in [4, 8, 12] or
        (12-dizhi) == 10 and a.lunarMonth in [2, 6, 10]):
        ZaYaoDetail('天巫')

    if ((12-dizhi) == 6 and a.lunarMonth in [1,11] or
        (12-dizhi) == 1 and a.lunarMonth == 2 or
        (12-dizhi) == 0 and a.lunarMonth == 3 or
        (12-dizhi) == 10 and a.lunarMonth in [4,9,12] or
        (12-dizhi) == 3 and a.lunarMonth in [5,8] or
        (12-dizhi) == 11 and a.lunarMonth == 6 or
        (12-dizhi) == 9 and a.lunarMonth ==7 or
        (12-dizhi) == 2 and a.lunarMonth ==10):
        ZaYaoDetail('天月')

    if ((12-dizhi) == 10 and a.lunarMonth in [1,7] or
    (12-dizhi) == 8 and a.lunarMonth in [2,8] or
    (12-dizhi) == 6 and a.lunarMonth in [3,9] or
    (12-dizhi) == 4 and a.lunarMonth in [4,10] or
    (12-dizhi) == 2 and a.lunarMonth in [5,11] or
    (12-dizhi) == 0 and a.lunarMonth in [6,12]):
        ZaYaoDetail('阴煞')

#----------安日系诸星
    if (a.twohourNum + (12-dizhi)-a.lunarDay+2)%12 == 6:
        ZaYaoDetail('恩光')
    if a.twohourNum == ((12-dizhi)-a.lunarDay+2)%12:
        ZaYaoDetail('天贵')

    if ((-a.lunarDay+1+(12-dizhi))%12 - a.lunarMonth) in [-1,11]:
        # (12-dizhi)+a.lunarDay-1
        ZaYaoDetail('三台')
    if ((a.lunarDay-1+(12-dizhi))%12 + a.lunarMonth) in [7,19]:
        ZaYaoDetail('八座')

#-----------安干系诸星
    if ((12-dizhi)==10 and a.year8Char[0]=='甲'
    or (12-dizhi)==11 and a.year8Char[0] == '乙'
    or (12-dizhi)==1 and a.year8Char[0] == '丙'
    or (12-dizhi)==2 and a.year8Char[0] == '丁'
    or (12-dizhi) == 1 and a.year8Char[0] == '戊'
    or (12-dizhi) == 2 and a.year8Char[0] == '己'
    or (12-dizhi) == 4 and a.year8Char[0] == '庚'
    or (12-dizhi)== 5 and  a.year8Char[0] == '辛'
    or (12-dizhi) == 7 and a.year8Char[0] == '壬'
    or (12-dizhi) == 8 and a.year8Char[0] == '癸'):
        ZaYaoDetail("禄存")

    if ((12-dizhi)==11 and a.year8Char[0]=='甲'
    or (12-dizhi)==0 and a.year8Char[0] == '乙'
    or (12-dizhi)==2 and a.year8Char[0] == '丙'
    or (12-dizhi)==3 and a.year8Char[0] == '丁'
    or (12-dizhi) == 2 and a.year8Char[0] == '戊'
    or (12-dizhi) == 3 and a.year8Char[0] == '己'
    or (12-dizhi) == 5 and a.year8Char[0] == '庚'
    or (12-dizhi)== 6 and  a.year8Char[0] == '辛'
    or (12-dizhi) == 8 and a.year8Char[0] == '壬'
    or (12-dizhi) == 9 and a.year8Char[0] == '癸'):
        ZaYaoDetail('擎羊')

    if ((12-dizhi)==9 and a.year8Char[0]=='甲'
    or (12-dizhi)==10 and a.year8Char[0] == '乙'
    or (12-dizhi)==0 and a.year8Char[0] == '丙'
    or (12-dizhi)==1 and a.year8Char[0] == '丁'
    or (12-dizhi) == 0 and a.year8Char[0] == '戊'
    or (12-dizhi) == 1 and a.year8Char[0] == '己'
    or (12-dizhi) == 3 and a.year8Char[0] == '庚'
    or (12-dizhi)== 4 and  a.year8Char[0] == '辛'
    or (12-dizhi) == 6 and a.year8Char[0] == '壬'
    or (12-dizhi) == 7 and a.year8Char[0] == '癸'):
        ZaYaoDetail("陀罗")

    if ((12-dizhi)==9 and a.year8Char[0]=='甲'
    or (12-dizhi)==8 and a.year8Char[0] == '乙'
    or (12-dizhi)==7 and a.year8Char[0] == '丙'
    or (12-dizhi)==7 and a.year8Char[0] == '丁'
    or (12-dizhi) == 9 and a.year8Char[0] == '戊'
    or (12-dizhi) == 8 and a.year8Char[0] == '己'
    or (12-dizhi) == 9 and a.year8Char[0] == '庚'
    or (12-dizhi)== 2 and  a.year8Char[0] == '辛'
    or (12-dizhi) == 11 and a.year8Char[0] == '壬'
    or (12-dizhi) == 11 and a.year8Char[0] == '癸'):
        ZaYaoDetail("天魁")

    if ((12-dizhi) == 3 and a.year8Char[0] == '甲'
    or (12-dizhi) == 4 and a.year8Char[0] == '乙'
    or (12-dizhi) == 5 and a.year8Char[0] == '丙'
    or (12-dizhi) == 5 and a.year8Char[0] == '丁'
    or (12-dizhi) == 3 and a.year8Char[0] == '戊'
    or (12-dizhi) == 4 and a.year8Char[0] == '己'
    or (12-dizhi) == 3 and a.year8Char[0] == '庚'
    or (12-dizhi) == 10 and a.year8Char[0] == '辛'
    or (12-dizhi) == 1 and a.year8Char[0] == '壬'
    or (12-dizhi) == 1 and a.year8Char[0] == '癸'):
        ZaYaoDetail("天钺")

    if ((12-dizhi)==3 and a.year8Char[0]=='甲'
    or (12-dizhi)==0 and a.year8Char[0] == '乙'
    or (12-dizhi)==1 and a.year8Char[0] == '丙'
    or (12-dizhi)==10 and a.year8Char[0] == '丁'
    or (12-dizhi) == 11 and a.year8Char[0] == '戊'
    or (12-dizhi) == 5 and a.year8Char[0] == '己'
    or (12-dizhi) == 7 and a.year8Char[0] == '庚'
    or (12-dizhi)== 5 and  a.year8Char[0] == '辛'
    or (12-dizhi) == 6 and a.year8Char[0] == '壬'
    or (12-dizhi) == 2 and a.year8Char[0] == '癸'):
        ZaYaoDetail("天官")

    if ((12-dizhi)==5 and a.year8Char[0]=='甲'
    or (12-dizhi)==4 and a.year8Char[0] == '乙'
    or (12-dizhi)==8 and a.year8Char[0] == '丙'
    or (12-dizhi)==7 and a.year8Char[0] == '丁'
    or (12-dizhi) == 11 and a.year8Char[0] == '戊'
    or (12-dizhi) == 10 and a.year8Char[0] == '己'
    or (12-dizhi) == 2 and a.year8Char[0] == '庚'
    or (12-dizhi)== 1 and  a.year8Char[0] == '辛'
    or (12-dizhi) == 2 and a.year8Char[0] == '壬'
    or (12-dizhi) == 1 and a.year8Char[0] == '癸'):
        ZaYaoDetail("天福")

#----------安年支诸星
# a.lunarYear%12（年支）:申0 酉1 戌2 亥3 子4 丑5 寅6 卯7 辰8 巳9 午10 未11
    if ((12 - dizhi) - a.lunarYear%12) in [-7, 5]:
        ZaYaoDetail('天空')
    if ((12 - dizhi) + a.lunarYear%12) in [6, 18]:
        ZaYaoDetail('天哭')
    if ((12 - dizhi) - a.lunarYear%12) in [-2, 10]:
        ZaYaoDetail('天虚')
    if ((12 - dizhi) - a.lunarYear%12) in [-4, 8]:
        ZaYaoDetail('龙池')
    if ((12 - dizhi) + a.lunarYear%12) in [10, 22]:
        ZaYaoDetail('凤阁')
    if ((12 - dizhi) + a.lunarYear%12) in [3,15] :
        ZaYaoDetail('红鸾')
    if ((12 - dizhi) + a.lunarYear%12) in [9, 21]:
        ZaYaoDetail('天喜')
    if ((12-dizhi)==4 and a.lunarYear%12 == 4
    or (12-dizhi)==5 and a.lunarYear%12 == 5
    or (12-dizhi)==6 and a.lunarYear%12 == 6
    or (12-dizhi)==1 and a.lunarYear%12 == 7
    or (12-dizhi) == 2 and a.lunarYear%12 == 8
    or (12-dizhi) == 3 and a.lunarYear%12 == 9
    or (12-dizhi) == 10 and a.lunarYear%12 == 10
    or (12-dizhi)== 11 and  a.lunarYear%12 == 11
    or (12-dizhi) == 0 and a.lunarYear%12 == 0
    or (12-dizhi) == 7 and a.lunarYear%12 == 1
    or (12 - dizhi) == 8 and a.lunarYear % 12 == 0
    or (12 - dizhi) == 9 and a.lunarYear % 12 == 1
    ):
        ZaYaoDetail('蜚廉')
    if ((12-dizhi)==1 and a.lunarYear%12 in [1,4,7,10]
    or (12-dizhi)==5 and a.lunarYear%12 in [2,5,8,11]
    or (12-dizhi)==9 and a.lunarYear%12 in [0,3,6,9]
    ):
        ZaYaoDetail('破碎')
    if ((12-dizhi)==10 and a.lunarYear%12 in [3,4,5]
    or (12-dizhi)==1 and a.lunarYear%12 in [6,7,8]
    or (12-dizhi)==4 and a.lunarYear%12 in [9,10,11]
    or (12-dizhi) ==7  and a.lunarYear % 12 in [0,1,2]
    ):
        ZaYaoDetail('孤辰')
    if ((12 - dizhi) == 2 and a.lunarYear % 12 in [3, 4, 5]
    or (12 - dizhi) == 5 and a.lunarYear % 12 in [6, 7, 8]
    or (12 - dizhi) == 8 and a.lunarYear % 12 in [9, 10, 11]
    or (12 - dizhi) == 11 and a.lunarYear % 12 in [0, 1, 2]
    ):
        ZaYaoDetail('寡宿')
    if (ind+a.lunarYear % 12) in [4,16]:
        ZaYaoDetail('天才')
    if (12-dizhi)==(shenDizhiNum+a.lunarYear % 12-4)%12:
        ZaYaoDetail('天寿')
    #------安天伤、天使
    if ind==7:
        ZaYaoDetail('天伤')
    elif ind==5:
        ZaYaoDetail('天使')