# coding:utf-8

import jieba
import re
import random
import pickle


def cut_word(text):
    restr = '[0-9\s+\.\!\/_,$%^*();?:\-<>《》【】+\"\']+|[+——！，；。？：、~@#￥%……&*（）]+'
    resu = text.replace('|', '').replace('&nbsp;', '').replace('ldquo', '').replace('rdquo',
                                                                                    '').replace(
        'lsquo', '').replace('rsquo', '').replace('“', '').replace('”', '').replace('〔', '').replace('〕', '')
    resu = re.split(r'\s+', resu)
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('', ''.join(resu))
    line = re.sub(restr, '', dd)
    seg_list = jieba.lcut(line)
    return seg_list


pos = [i for i in open('DATA/pos.txt', 'r', encoding='utf8')]
neg = [i for i in open('DATA/neg.txt', 'r', encoding='utf8')]
xs = list()
ys = list()
for i in range(len(pos)):
    xs.append(" ".join(cut_word(pos[i])))
    ys.append("1")
    if i % 100 == 0:
        print(i / 100)
for i in range(len(neg)):
    xs.append(' '.join(cut_word(neg[i])))
    ys.append("0")
    if i % 100 == 0:
        print(i / 100)
train = []
label = []

id = list(range(len(ys)))
random.shuffle(id)
f = open("DATA/train.txt", "w", encoding="utf-8")
g = open("DATA/label.txt", "w", encoding="utf-8")
for i in id:
    f.write(xs[i] + "\n")
    g.write(ys[i] + "\n")
f.close()
g.close()

pass
