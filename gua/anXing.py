import datetime
import cnlunar

# a = cnlunar.Lunar(datetime.datetime(1993, 1, 21, 18), godType='8char')  # 常规算法
# a = cnlunar.Lunar(datetime.datetime(2022, 2, 3, 10, 30), godType='8char', year8Char='beginningOfSpring')  # 八字立春切换算法
# a.gender = "男"

birthday=input('请输入公历出生日期(Y/M/D/H，例如：1994061602):')
a = cnlunar.Lunar(datetime.datetime(int(birthday[0:4]),int(birthday[4:6]),int(birthday[6:8]),int(birthday[8:10])))
a.gender = input('请输入性别：')
dic = {
    '性别': a.gender,
    '日期': a.date,
    '农历数字': (a.lunarYear, a.lunarMonth, a.lunarDay, '闰' if a.isLunarLeapMonth else ''),
    '农历': '%s %s[%s]年 %s%s' % (a.lunarYearCn, a.year8Char, a.chineseYearZodiac, a.lunarMonthCn, a.lunarDayCn),
    '八字': ' '.join([a.year8Char, a.month8Char, a.day8Char, a.twohour8Char]),
    '今日五行': a.get_today5Elements(),

}
for i in dic:
    midstr = '\t' * (2 - len(i) // 2) + ':' + '\t'
    print(i, midstr, dic[i])

# print('年干：',a.year8Char[0],'，年支：',a.year8Char[1],"，时间数：" , a.twohourNum)


TianganList2=['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
# TianganList2=['甲','癸','壬','辛','庚','己','戊','丁','丙','乙']
DizhiList=['辰','巳','午','未','申','酉','戌','亥','子','丑']
# twohourNum ：子时0 丑1 寅2 卯3 辰4巳5午6未7 申8酉9戌10亥11
# 安命宫
for i in range(0, 12):
    if a.isLunarLeapMonth:
        a.lunarMonth +=1
    if a.lunarMonth % 12 == (a.twohourNum + i + 3) % 12:
        DizhiNum = i
    if a.lunarMonth % 12 == (a.twohourNum + i -3) % 12:
        shenDizhiNum = i
# print('命宫地支数',DizhiNum,'身宫地支数',shenDizhiNum)


# 安十二宫天干 甲0乙1丙2丁3戊4己5庚6辛7壬8癸9
# 此时j为辰宫地支对应天干数
if a.year8Char[0] == '甲' or a.year8Char[0] == '己':
        TianganList=['戊','己','庚','辛','壬','癸','甲','乙','丙','丁','丙','丁']
if a.year8Char[0] == '乙' or a.year8Char[0] == '庚':
        TianganList=['庚','辛','壬','癸','甲','乙','丙','丁','戊','己','戊','己']
if a.year8Char[0] == '丙' or a.year8Char[0] == '辛':
        TianganList=['壬','癸','甲','乙','丙','丁','戊','己','庚','辛','庚','辛',]
if a.year8Char[0] == '丁' or a.year8Char[0] == '壬':
        TianganList=['甲','乙','丙','丁','戊','己','庚','辛','壬','癸','壬','癸']
if a.year8Char[0] == '戊' or a.year8Char[0] == '癸':
        TianganList=['丙','丁','戊','己','庚','辛','壬','癸','甲','乙','甲','乙']
# print(TianganList[j], DizhiList[DizhiNum])
# if a.year8Char[0] == '丁' or a.year8Char[0] == '壬':
#     if a.twohour8Char[1] == '酉':
#         if a.lunarMonth == 12:
#             print("命宫地支为辰")
#             MingTiangan = '甲'
#             print("命宫天干为甲")

if a.year8Char[0] == '甲' or a.year8Char[0] == '己':
    if DizhiNum in [6, 7]:
        #戌亥
        Wuxingju = '火六局'
    elif DizhiNum in [8, 9]:
        Wuxingju = '水二局'
    elif DizhiNum in [2, 3]:
        Wuxingju = '土五局'
    elif DizhiNum in [4, 5]:
        Wuxingju = '金四局'
    elif DizhiNum in [0, 1]:
        Wuxingju = '木三局'
    else:
        Wuxingju = '火六局'
elif a.year8Char[0] == '乙' or a.year8Char[0] == '庚':
    if DizhiNum in [8, 9]:
        # if MingDizhi == '辰' or MingDizhi =='巳':
        Wuxingju = '火六局'
    elif DizhiNum in [4, 5]:
        Wuxingju = '水二局'
    elif DizhiNum in [10, 11]:
        Wuxingju = '土五局'
    elif DizhiNum in [0, 1]:
        Wuxingju = '金四局'
    elif DizhiNum in [2, 3]:
        Wuxingju = '木三局'
    else:
        Wuxingju = '土五局'
elif a.year8Char[0] == '丙' or a.year8Char[0] == '辛':
    if DizhiNum in [4, 5]:
        # if MingDizhi == '辰' or MingDizhi =='巳':
        Wuxingju = '火六局'
    elif DizhiNum in [0, 1]:
        Wuxingju = '水二局'
    elif DizhiNum in [8, 9]:
        Wuxingju = '土五局'
    elif DizhiNum in [2, 3]:
        Wuxingju = '金四局'
    elif DizhiNum in [10, 11]:
        Wuxingju = '木三局'
    else:
        Wuxingju = '木三局'
elif a.year8Char[0] == '丁' or a.year8Char[0] == '壬':
    if DizhiNum in [0, 1]:
        # if MingDizhi == '辰' or MingDizhi =='巳':
        Wuxingju = '火六局'
    elif DizhiNum in [2, 3]:
        Wuxingju = '水二局'
    elif DizhiNum in [4, 5]:
        Wuxingju = '土五局'
    elif DizhiNum in [6, 7]:
        Wuxingju = '金四局'
    elif DizhiNum in [8, 9]:
        Wuxingju = '木三局'
    else:
        Wuxingju = '水二局'
# a.year8Char[0] == '戊' or a.year8Char[0] == '癸'
else:
    if DizhiNum in [2, 3]:
        # if MingDizhi == '辰' or MingDizhi =='巳':
        Wuxingju = '火六局'
    elif DizhiNum in [6, 7]:
        Wuxingju = '水二局'
    elif DizhiNum in [0, 1]:
        Wuxingju = '土五局'
    elif DizhiNum in [8, 9]:
        Wuxingju = '金四局'
    elif DizhiNum in [4, 5]:
        Wuxingju = '木三局'
    else:
        Wuxingju = '水二局'
print(Wuxingju)

# 安紫微星
if Wuxingju == '水二局':
    if a.lunarDay in [2,3, 26,27]:
        ZiweixingPos = 1
    elif a.lunarDay in [14,15]:
        ZiweixingPos = 2
    elif a.lunarDay in [8,9]:
        ZiweixingPos = 3
    elif a.lunarDay in [20,21]:
        ZiweixingPos = 4
    elif a.lunarDay in [6,7,30]:
        ZiweixingPos = 5
    elif a.lunarDay in [18,19]:
        ZiweixingPos = 6
    elif a.lunarDay in [1,24,25]:
        ZiweixingPos = 7
    elif a.lunarDay in [12,13]:
        ZiweixingPos = 8
    elif a.lunarDay in [22,23]:
        ZiweixingPos = 9
    elif a.lunarDay in [10,11]:
        ZiweixingPos = 10
    elif a.lunarDay in [4,5,28,29]:
        ZiweixingPos = 11
    else:
        ZiweixingPos = 12
if Wuxingju == '木三局':
    if a.lunarDay in [3,5]:
        ZiweixingPos = 1
    elif a.lunarDay in [13,21]:
        ZiweixingPos = 2
    elif a.lunarDay in [4,12,14]:
        ZiweixingPos = 3
    elif a.lunarDay in [22,30]:
        ZiweixingPos = 4
    elif a.lunarDay in [1,9,11]:
        ZiweixingPos = 5
    elif a.lunarDay in [19,27,29]:
        ZiweixingPos = 6
    elif a.lunarDay in [2,28]:
        ZiweixingPos = 7
    elif a.lunarDay in [18,20]:
        ZiweixingPos = 8
    elif a.lunarDay in [25]:
        ZiweixingPos = 9
    elif a.lunarDay in [7,15,17]:
        ZiweixingPos = 10
    elif a.lunarDay in [6,8]:
        ZiweixingPos = 11
    else:
        ZiweixingPos = 12
if Wuxingju == '金四局':
    if a.lunarDay in [4, 7,13]:
        ZiweixingPos = 1
    elif a.lunarDay in [18, 28]:
        ZiweixingPos = 2
    elif a.lunarDay in [6, 16, 19,25]:
        ZiweixingPos = 3
    elif a.lunarDay in [1, 39]:
        ZiweixingPos = 4
    elif a.lunarDay in [2, 12, 15,21]:
        ZiweixingPos = 5
    elif a.lunarDay in [26]:
        ZiweixingPos = 6
    elif a.lunarDay in [3, 9]:
        ZiweixingPos = 7
    elif a.lunarDay in [14, 24, 27]:
        ZiweixingPos = 8
    elif a.lunarDay in [5]:
        ZiweixingPos = 9
    elif a.lunarDay in [10, 20, 23,29]:
        ZiweixingPos = 10
    elif a.lunarDay in [8,11,17]:
        ZiweixingPos = 11
    else:
        ZiweixingPos = 12
if Wuxingju == '土五局':
    if a.lunarDay in [5, 9,17]:
        ZiweixingPos = 1
    elif a.lunarDay in [11, 23]:
        ZiweixingPos = 2
    elif a.lunarDay in [8,20,24]:
        ZiweixingPos = 3
    elif a.lunarDay in [2,26]:
        ZiweixingPos = 4
    elif a.lunarDay in [3,15,19,27]:
        ZiweixingPos = 5
    elif a.lunarDay in [6,18,30]:
        ZiweixingPos = 6
    elif a.lunarDay in [4,12]:
        ZiweixingPos = 7
    elif a.lunarDay in [6,18,30]:
        ZiweixingPos = 8
    elif a.lunarDay in [7]:
        ZiweixingPos = 9
    elif a.lunarDay in [1,13,25,29]:
        ZiweixingPos = 10
    elif a.lunarDay in [10,14,22]:
        ZiweixingPos = 11
    else:
        ZiweixingPos = 12
if Wuxingju == '火六局':
    if a.lunarDay in [6, 11, 21]:
        ZiweixingPos = 1
    elif a.lunarDay in [14, 28]:
        ZiweixingPos = 2
    elif a.lunarDay in [10, 24, 29]:
        ZiweixingPos = 3
    elif a.lunarDay in [3, 13]:
        ZiweixingPos = 4
    elif a.lunarDay in [4, 18, 23]:
        ZiweixingPos = 5
    elif a.lunarDay in [7, 26]:
        ZiweixingPos = 6
    elif a.lunarDay in [5, 15, 25]:
        ZiweixingPos = 7
    elif a.lunarDay in [8, 22]:
        ZiweixingPos = 8
    elif a.lunarDay in [9, 19]:
        ZiweixingPos = 9
    elif a.lunarDay in [2, 16, 30]:
        ZiweixingPos = 10
    elif a.lunarDay in [12, 17]:
        ZiweixingPos = 11
    else:
        ZiweixingPos = 12

#------------选定基本盘
#紫微寅申
if ZiweixingPos == 1 or ZiweixingPos == 2:
    star1st = "贪狼"
    star2nd = "太阴"
    star3rd = "紫微天府"
    star4th = "天机"
    star5th = "破军"
    star6th = "太阳"
    star7th = "武曲"
    star8th = "天同"
    star9th = "七杀"
    star10th = "天梁"
    star11th = "廉贞天相"
    star12th = "巨门"
# 巳亥
elif ZiweixingPos == 3 or ZiweixingPos == 4:
    star1st = "天机天梁"
    star2nd = "天相"
    star3rd = "太阳巨门"
    star4th = "武曲贪狼"
    star5th = "天同太阴"
    star6th = "天府"
    star7th = "空宫"
    star8th = "廉贞破军"
    star9th = "空宫"
    star10th = "空宫"
    star11th = "空宫"
    star12th = "紫微七杀"
# 辰戌
elif ZiweixingPos in [5,6]:
    star1st = "紫微天相"
    star2nd = "天机巨门"
    star3rd = "贪狼"
    star4th = "太阳太阴"
    star5th = "武曲天府"
    star6th = "天同"
    star7th = "破军"
    star8th = "空宫"
    star9th = "廉贞"
    star10th = "空宫"
    star11th = "七杀"
    star12th = "天梁"
elif ZiweixingPos in [7,8]:
# 丑未
    star1st = "太阴"
    star2nd = "天府"
    star3rd = "空宫"
    star4th = "紫微破军"
    star5th = "天机"
    star6th = "空宫"
    star7th = "太阳"
    star8th = "武曲七杀"
    star9th = "天同天梁"
    star10th = "天相"
    star11th = "巨门"
    star12th = "廉贞贪狼"
# 子午
elif ZiweixingPos == 9 or ZiweixingPos == 10:
    star1st = "廉贞天府"
    star2nd = "空宫"
    star3rd = "破军"
    star4th = "空宫"
    star5th = "紫微"
    star6th = "天机"
    star7th = "七杀"
    star8th = "太阳天梁"
    star9th = "武曲天相"
    star10th = "天同巨门"
    star11th = "贪狼"
    star12th = "太阴"
# 卯酉
else :
    star1st = "巨门"
    star2nd = "紫微贪狼"
    star3rd = "天机太阴"
    star4th = "天府"
    star5th = "太阳"
    star6th = "武曲破军"
    star7th = "天同"
    star8th = "空宫"
    star9th = "空宫"
    star10th = "廉贞七杀"
    star11th = "天梁"
    star12th = "天相"
#---------区分并唯一确定紫微星位置
if ZiweixingPos%2 == 1:
    starlist =[star1st, star2nd, star3rd, star4th, star5th, star6th, star7th, star8th, star9th, star10th, star11th, star12th]
else:
    starlist = [star7th, star8th, star9th, star10th, star11th, star12th,star1st, star2nd, star3rd, star4th, star5th, star6th]
#----------这个地支顺序最好用顺时针，否则安星法的地支变量需要替换，此处再需调整
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

# a_lunar_year = a.lunarYear % 12

