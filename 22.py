import pymongo
import decimal

myclient = pymongo.MongoClient("mongodb://intelligent:intelligent@172.16.11.199:27017/operation")
mydb = myclient['operation']
mycol = mydb['clues']

print('PILFlag: {}'.format(mycol.find({'clue.PILFlag': {'$exists': True}}).count()))
print('caseTypeFlag: {}'.format(mycol.find({'clue.caseTypeFlag': {'$exists': True}}).count()))

pil = 0
caseType = 0
count = 0
for x in mycol.find(
        { 'clue.PILFlag':{'$exists': True},'clue.caseTypeFlag':{'$exists': True},'passFlag':{'$exists': True},'operationDate':{"$gt":1544803200000}}):
    # if x.get('passFlag') == x.get('clue').get('PILFlag'):
    #     pil += 1
    # if x.get('passFlag') == x.get('clue').get('caseTypeFlag'):
    #     caseType += 1

    # if 1 == x.get('clue').get('PILFlag'):
    #     pil += 1
    # if 1 == x.get('clue').get('caseTypeFlag'):
    #     caseType += 1
    if x.get('clue').get('PILFlag') == x.get('passFlag'):
        pil += 1
    if x.get('clue').get('caseTypeFlag') == x.get('passFlag'):
        caseType += 1
    if x.get('passFlag') in [0,1]:
        count += 1
print('count: {}, pil: {}, acc: {}, caseType: {}, acc: {}'.format(count, pil, float(
    decimal.Decimal(pil / count).quantize(decimal.Decimal('0.0000'))), caseType, float(
    decimal.Decimal(caseType / count).quantize(decimal.Decimal('0.0000')))))
#
# pil = 0
# count = 0
# for x in mycol.find({'clue.PILFlag': {'$exists': True}, 'passFlag': {'$in': [0, 1]}}):
#     if x.get('passFlag') == x.get('clue').get('PILFlag'):
#         pil += 1
#     count += 1
#
# print('count: {}, pil: {}, acc: {}'.format(count, pil,
#                                            float(decimal.Decimal(pil / count).quantize(decimal.Decimal('0.0000')))))
