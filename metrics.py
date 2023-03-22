from sklearn.metrics import confusion_matrix

def CM(label, pred):
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]

class _metrics():
    def __init__(self, pred, label):
        super().__init__()
        self.pred = pred
        self.label = label
        self.TP, self.FN, self.FP, self.TN = CM(self.pred,self.label)

    def acc(self):
        acc = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN + 1e-5)
        return round(acc,4)

    def ppv(self):
        ppv = self.TP/(self.TP+self.FP+ 1e-5)
        return round(ppv,4)

    def tpr(self):
        tpr = (self.TP) / (self.TP + self.FN+ 1e-5)
        return round(tpr,4)

    def tnr(self):
        tnr = (self.TN)/(self.TN+self.FP + 1e-5)
        return round(tnr,4)

    def cm(self):
        return int(self.TP), int(self.FN), int(self.FP), int(self.TN)

def metrics_(pred, label, metrics):
    result_c_performance = []
    tfpn_calculate = {}
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            tfpn_calculate[metric] = _metrics(pred, label).acc()
        elif metric == 'ppv':
            tfpn_calculate[metric] = _metrics(pred, label).ppv()
        elif metric == 'tpr':
            tfpn_calculate[metric] = _metrics(pred, label).tpr()
        elif metric == 'tnr':
            tfpn_calculate[metric] = _metrics(pred, label).tnr()
        elif metric == 'cm':
            TP, FN, FP, TN = _metrics(pred, label).cm()
            confusion_matrix = dict({'tp':TP,'fn':FN,'fp':FP,'tn':TN})
        else:
            raise ValueError('metric %s not recognized' % metric)
    result_c_performance.append(confusion_matrix)
    result_c_performance.append(tfpn_calculate)

    return result_c_performance

