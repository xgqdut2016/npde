from anXing import *
from SihuaDetail import *
import re
#忘记加入左辅右弼、文昌文曲四化
def birthSihua(star,ind,year):
    if year == '甲':
        if re.search(r"廉贞",star):
            print(gongname[ind],"廉贞化禄")
            SihuaDetail('廉贞化禄',ind)
        if re.search(r"破军",star):
            print(gongname[ind],"破军化权")
            SihuaDetail('破军化权', ind)
        if re.search(r"武曲",star):
            print(gongname[ind],"武曲化科")
            SihuaDetail('武曲化科', ind)
        if re.search(r"太阳",star):
            print(gongname[ind],"太阳化忌")
            SihuaDetail('太阳化忌', ind)
    elif year == '乙':
        # print("天机化禄，天梁化权，紫微化科，太阴化忌")
        if re.search(r"天机",star):
            print(gongname[ind],"天机化禄")
            SihuaDetail('天机化禄', ind)
        if re.search(r"天梁",star):
            print(gongname[ind],"天梁化权")
            SihuaDetail('天梁化权', ind)
        if re.search(r"紫微",star):
            print(gongname[ind],"紫微化科")
            SihuaDetail('紫微化科', ind)
        if re.search(r"太阴",star):
            print(gongname[ind],"太阴化忌")
            SihuaDetail('太阴化忌', ind)
    elif year == '丙':
        # print("天同化禄，天机化权，文昌化科，廉贞化忌")
        if re.search(r"天同",star):
            print(gongname[ind],"天同化禄")
            SihuaDetail('天同化禄', ind)
        if re.search(r"天机",star):
            print(gongname[ind],"天机化权")
            SihuaDetail('天机化权', ind)
        if re.search(r"文昌",star):
            print(gongname[ind],"文昌化科")
            SihuaDetail('文昌化科', ind)
        if re.search(r"廉贞",star):
            print(gongname[ind],"廉贞化忌")
            SihuaDetail('廉贞化忌', ind)
    elif year == '丁':
        if re.search(r"太阴",star):
            print(gongname[ind],"太阴化禄")
            SihuaDetail('太阴化禄', ind)
        if re.search(r"天同",star):
            print(gongname[ind],"天同化权")
            SihuaDetail('天同化权', ind)
        if re.search(r"天机",star):
            print(gongname[ind],"天机化科")
            SihuaDetail('天机化科', ind)
        if re.search(r"巨门",star):
            print(gongname[ind],"巨门化忌")
            SihuaDetail('巨门化忌', ind)
    elif year == '戊':
        # print("贪狼化禄，太阴化权，右弼化科，天机化忌")
        if re.search(r"贪狼",star):
            print(gongname[ind],"贪狼化禄")
            SihuaDetail('贪狼化禄', ind)
        if re.search(r"太阴",star):
            print(gongname[ind],"太阴化权")
            SihuaDetail('太阴化权', ind)
        if re.search(r"右弼",star):
            print(gongname[ind],"右弼化科")
            SihuaDetail('右弼化科', ind)
        if re.search(r"天机",star):
            print(gongname[ind],"天机化忌")
            SihuaDetail('天机化忌', ind)
    elif year == '己':
        # print("武曲化禄，贪狼化权，天梁化科，文曲化忌")
        if re.search(r"武曲",star):
            print(gongname[ind],"武曲化禄")
            SihuaDetail('武曲化禄', ind)
        if re.search(r"贪狼",star):
            print(gongname[ind],"贪狼化权")
            SihuaDetail('贪狼化权', ind)
        if re.search(r"天梁",star):
            print(gongname[ind],"天梁化科")
            SihuaDetail('天梁化科', ind)
        if re.search(r"文曲",star):
            print(gongname[ind],"文曲化忌")
            SihuaDetail('文曲化忌', ind)
    elif year == '庚':
        # print("太阳化禄，武曲化权，太阴化科，天同化忌")
        if re.search(r"太阳",star):
            print(gongname[ind],"太阳化禄")
            SihuaDetail('太阳化禄', ind)
        if re.search(r"武曲",star):
            print(gongname[ind],"武曲化权")
            SihuaDetail('武曲化权', ind)
        if re.search(r"太阴",star):
            print(gongname[ind],"太阴化科")
            SihuaDetail('太阴化科', ind)
        if re.search(r"天同",star):
            print(gongname[ind],"天同化忌")
            SihuaDetail('天同化忌', ind)
    elif year == '辛':
        # print("巨门化禄，太阳化权，文曲化科，文昌化忌")
        if re.search(r"巨门",star):
            print(gongname[ind],"巨门化禄")
            SihuaDetail('巨门化禄', ind)
        if re.search(r"太阳",star):
            print(gongname[ind],"太阳化权")
            SihuaDetail('太阳化权', ind)
        if re.search(r"文曲",star):
            print(gongname[ind],"文曲化科")
            SihuaDetail('文曲化科', ind)
        if re.search(r"文昌",star):
            print(gongname[ind],"文昌化忌")
            SihuaDetail('文昌化忌', ind)
    elif year == '壬':
        # print("天梁化禄，紫微化权，左辅化科，武曲化忌")
        if re.search(r"天梁",star):
            print(gongname[ind],"天梁化禄")
            SihuaDetail('天梁化禄', ind)
        if re.search(r"紫微",star):
            print(gongname[ind],"紫微化权")
            SihuaDetail('紫微化权', ind)
        if re.search(r"左辅",star):
            print(gongname[ind],"左辅化科")
            SihuaDetail('左辅化科', ind)
        if re.search(r"武曲",star):
            print(gongname[ind],"武曲化忌")
            SihuaDetail('武曲化忌', ind)
    elif year == '癸':
        # print("破军化禄，巨门化权，太阴化科，贪狼化忌")
        if re.search(r"破军",star):
            print(gongname[ind],"破军化禄")
            SihuaDetail('破军化禄', ind)
        if re.search(r"巨门",star):
            print(gongname[ind],"巨门化权")
            SihuaDetail('巨门化权', ind)
        if re.search(r"太阴",star):
            print(gongname[ind],"太阴化科")
            SihuaDetail('太阴化科', ind)
        if re.search(r"贪狼",star):
            print(gongname[ind],"贪狼化忌")
            SihuaDetail('贪狼化忌', ind)