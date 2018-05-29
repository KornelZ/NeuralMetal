from pymining import seqmining
import math

class SequenceMining(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        src = []
        for i in range(0, len(source) - 8):
            src.append(source[i:i + 8])
        #relim_input_source = itemmining.get_relim_input(source)
        self.sourcereport = seqmining.freq_seq_enum(src, min_support=4)
        trg = []
        for i in range(0, len(target) - 8):
            trg.append(target[i:i + 8])
        #relim_input_target = itemmining.get_relim_input(target)
        self.targetreport = seqmining.freq_seq_enum(trg, min_support=4)

    def _count_sum(self, sample, report):
        sum = 0
        min_length = 8
        sequences = [seq for seq, _ in report if len(seq) >= min_length]
        sequences.sort(key=len, reverse=True)
        i = 0
        while i < len(sample):
            success = False
            for seq in sequences:
                if i + len(seq) < len(sample) and list(seq) == sample[i:i + len(seq)]:
                    i += len(seq)
                    sum += len(seq)
                    success = True
                    break
            if not success:
                i += 1
        return sum

    def calculate(self):
        sourcesum = self._count_sum(self.source, self.sourcereport)
        targetsum = self._count_sum(self.target, self.targetreport)
        return sourcesum / len(self.source), targetsum / len(self.target)


