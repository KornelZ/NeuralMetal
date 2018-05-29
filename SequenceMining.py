from pymining import seqmining
import math

class SequenceMining(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        #relim_input_source = itemmining.get_relim_input(source)
        self.sourcereport = seqmining.freq_seq_enum(source, min_support=2)
        #relim_input_target = itemmining.get_relim_input(target)
        self.targetreport = seqmining.freq_seq_enum(target, min_support=2)

    def _count_sum(self, sample, report, depth):
        sum = 0
        for i in range(len(sample)):
            value = []
            depth = min(depth, len(sample) - i)
            for j in range(depth):
                value.append(sample[i + j].replace(".0", ""))
            flag = False
            for j in range(depth):
                for key in report:
                    if key[0] == tuple(value):
                        sum = sum + len(value)
                        i = i + len(value)
                        flag = True
                        break
                if flag:
                    break
                value = value[:-1]
        return sum

    def calculate(self, depth):
        sourcesum = self._count_sum(self.source, self.sourcereport, depth)
        targetsum = self._count_sum(self.target, self.targetreport, depth)
        return sourcesum / len(self.source), targetsum / len(self.target)


