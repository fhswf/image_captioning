from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

annFile = "coco/annotations/captions_val2014.json"
resFile = "res.json"

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

imgIds = cocoRes.getImgIds()

res = {}
for imgId in imgIds:
    res[imgId] = cocoRes.imgToAnns[imgId]

for id in imgIds:
    if len(res[id]) != 1:
        print("Warning: id: {}: len={}".format(id, len(res[id])))
        cocoRes.imgToAnns[id] = res[id][0:1]

cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()

def getCIDEr(id):
    return cocoEval.imgToEval[id]['CIDEr']

imgIds.sort(key = getCIDEr)

num = 100

for id in imgIds[0 : num]:
    print('ID: {}, CIDEr: {}'.format(id, cocoEval.imgToEval[id]['CIDEr']))
for id in imgIds[len(imgIds) - num : len(imgIds)]:
    print('ID: {}, CIDEr: {}'.format(id, cocoEval.imgToEval[id]['CIDEr']))