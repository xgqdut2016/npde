#辅佐八曜
fuyao = ['天魁','天铖','左辅','右弼']
zuoyao = ['文昌','文曲','禄存','天马']
shayao = ['擎羊', '陀罗', '火星', '铃星','地空','地劫']#四大煞曜+空劫=煞曜
huayao = ['化禄', '化科', '化权','化忌']#化曜

kongyao = ['天空','地空', '地劫','截空','旬空']#空曜

xinyao = ['天刑','擎羊']#刑曜

jiyao = ['化忌','陀罗']#忌曜

taohua = ['咸池', '红鸾', '天喜','沐浴',  '天姚' ,'大耗']#桃花曜
tian_good = ['天德','天寿','天贵','天官','天福','天巫','天才']#天开头吉曜
tian_bad = ['天伤', '天哭','天虚','天月']#天开头煞曜


caiyi = ['三台', '八座', '龙池','凤阁','天才','封诰','华盖','台辅']#才艺官运曜

za_good = ['恩光','解神','月德','长生','冠带','临官','帝旺']#杂吉曜
za_bad = ['蜚廉','孤辰', '寡宿', '破碎','阴煞']#杂坏曜

def printsmallStar(fuyaoList,zuoyaoList,
                   shayaoList,huayaoList):
    smallStarList = []
    if(len(fuyaoList) > 0):
        print("辅曜：", end = ' ')
        for smallStar in fuyaoList:
            smallStarList.append(smallStarList)
            print(smallStar, end = ' ')
        print("\n")
    if(len(zuoyaoList) > 0):
        print("佐曜：", end = ' ')
        for smallStar in zuoyaoList:
            smallStarList.append(smallStarList)
            print(smallStar, end = ' ')
        print("\n")
    if(len(shayaoList) > 0):
        print("煞曜：", end = ' ')
        for smallStar in shayaoList:
            smallStarList.append(smallStarList)
            print(smallStar, end = ' ')
        print("\n")
    
    if(len(huayaoList) > 0):
        print("化曜：", end = ' ')
        for smallStar in huayaoList:
            smallStarList.append(smallStarList)
            print(smallStar, end = ' ')
        print("\n")
    
    print('---------------------------------------------')
    return smallStarList