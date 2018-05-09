import glob
from preprocess import parse_song
from SequenceMining import SequenceMining as Sequence
from StringDistance import StringDistance as Distance
import csv

class Report(object):

    def generate_report(self):
        targetf = []
        sourcef = []
        for file in glob.glob("midi_songs/**/*.mid", recursive=True):
            targetf.append(file)
        for file in glob.glob("output/**/*.mid", recursive=True):
            sourcef.append(file)
        source = []
        target = []
        for i in range(len(sourcef)):
            song = []
            parse_song(sourcef[i], song)
            source.append(song)
        for i in range(len(targetf)):
            song = []
            parse_song(targetf[i], song)
            target.append(song)

        with open("report.csv", "w") as f:
            wr = csv.writer(f)
            wr.writerow(["source", "target", "% nut w sequence", "levensteinh distance"])
            for sourcefile, sourcepath in zip(source, sourcef):
                for targetfile, targetpath in zip(target, targetf):
                    seqcalc = Sequence(sourcefile, targetfile)
                    result = seqcalc.calculate(3)
                    print(result)
                    calc = Distance(sourcefile, targetfile)
                    distance = calc.calculate()
                    print(distance)
                    wr.writerow([sourcepath, targetpath, result, distance])