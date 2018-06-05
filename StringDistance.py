import numpy

class StringDistance(object):

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def calculate(self):
        width = len(self.source)
        heigth = len(self.target)
        sum = max(width, heigth)
        table = numpy.zeros((width, heigth))
        for i in range(width):
            for j in range(heigth):
                cost = 1
                if self.source[i] == self.target[j]:
                    cost = 0
                above = table[i-1, j] + 1
                left = table[i, j-1] + 1
                diagonal = table[i-1, j-1] + cost
                table[i, j] = min(above, left, diagonal)
        if width < 1 or heigth < 1:
            return 0
        return table[width -1][heigth -1]/sum
