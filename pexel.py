class PEXEL:
    def __init__(self, annotations_file):
        print("Loading captions from pexels dataset ...")
        self.annotations_file = annotations_file
        self.dataset = dict()
        self.anns = dict()
        if not annotations_file == None:
            self.dataset = json.load(open(annotations_file, 'r'))
        self.createIndex()

    def createIndex(self):
        anns = {}
        for entry in self.dataset:
            anns[int(entry['_id'])] = entry['annotation']
        self.anns = anns
        print('pexels: loaded {} captions'.format(len(anns)))

    def getImgPath(self, id):
        return 'img_{}.jpg'.format(id)