import glob
from preprocess import parse_song
from SequenceMining import SequenceMining as Sequence
from StringDistance import StringDistance as Distance
import numpy as np
import csv


class Report(object):
    """Create report in report directory"""
    def generate_report(self, config):
        targetf = []
        sourcef = []
        for file in glob.glob("midi_songs/**/*.mid", recursive=True):
            sourcef.append(file)
        for file in glob.glob("output/**/*.mid", recursive=True):
            targetf.append(file)
        source = []
        target = []
        for i in range(len(sourcef)):

            _, song = parse_song(sourcef[i], config)
            source.append(song)
        for i in range(len(targetf)):
            _, song = parse_song(targetf[i], config)
            target.append(song)

        with open("reports/report_" + config.MODEL_NAME + ".csv", "w") as f:
            distances = []
            sqDistancesSrc = []
            sqDistancesTrg = []
            wr = csv.writer(f)
            wr.writerow(["source", "target", "procent nut w sekwencjach orginalu", "", "procent nut w sekwencjach nowego utworu", "", "levensteinh distance"])
            for sourcefile, sourcepath in zip(source, sourcef):
                for targetfile, targetpath in zip(target, targetf):
                    seqcalc = Sequence(sourcefile, targetfile)
                    result = seqcalc.calculate()
                    print(result)
                    calc = Distance(sourcefile, targetfile)
                    distance = calc.calculate()
                    print(distance)
                    distances.append(distance)
                    sqDistancesSrc.append(result[0])
                    sqDistancesTrg.append(result[1])
                    wr.writerow([sourcepath, targetpath, result[0], "",  result[1], "", distance])
            wr.writerow(["", "sredni procent nut w sekwencjach orginalu", np.mean(sqDistancesSrc), "sredni procent nut w sekwencjach nowego utworu", np.mean(sqDistancesTrg), "sredni procent dystansu Levenshteina", np.mean(distances)])
            wr.writerow(["", "standardowe odchylenie procentu nut w sekwencjach orginalu", np.std(sqDistancesSrc), "standardowe odchylenie nowego utworu", np.std(sqDistancesTrg), "standardowe odchylenie dystansu Levenshteina", np.std(distances)])
