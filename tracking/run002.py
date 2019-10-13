import gpu_tracker000 as t0
import gpu_tracker002 as t2
import os
import cv2

class args_:
    def __init__(self, seq, json, savefig, display):
        self.seq = seq
        self.json = json
        self.savefig = savefig
        self.display = display

a = os.path.realpath(__file__)
b = a.split('\\')
c = b[:len(b)-2]
d = ''
for i in range(len(c)):
    d += c[i]
    if i < len(c)-1: d += '\\'
print(d)
e = os.listdir(d + '/datasets/OTB')
print(e)

FW = 'model\tlength\tT2M0\tT2M1\n'
file_ = open('result.txt', 'w')
file_.write(FW)
file_.close()

for i in range(len(e)):

    args = args_(e[i], '', True, True)

    # using tracker 002 model 000
    f = open('R_' + e[i] + '_T2_M0.txt', 'w')
    overlapT2M0, trackerName = t2.main(args, 'models/model000.pth')
    writeText = ''
    for j in range(len(overlapT2M0)): writeText += str(overlapT2M0[j]) + '\n'
    writeText += '\nMean: ' + str(overlapT2M0.mean())
    f.write(writeText)
    f.close()

    # using tracker 002 model 001
    f = open('R_' + e[i] + '_T2_M1.txt', 'w')
    overlapT2M1, trackerName = t2.main(args, 'models/model001.pth')
    writeText = ''
    for j in range(len(overlapT2M1)): writeText += str(overlapT2M1[j]) + '\n'
    writeText += '\nMean: ' + str(overlapT2M1.mean())
    f.write(writeText)
    f.close()

    DatasetResult = e[i] + '\t' + str(len(overlapT2M0)) + '\t' + str(round(overlapT2M0.mean(), 6)) + '\t' + str(round(overlapT2M1.mean(), 6)) + '\n'
    FW = FW + DatasetResult

    file_ = open('result.txt', 'w')
    file_.write(FW)
    file_.close()
