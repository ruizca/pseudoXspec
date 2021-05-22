from enum import Enum


class Detector(Enum):
    M1 = ("EMOS1", "mos")
    M2 = ("EMOS2", "mos")
    PN = ("EPN", "pn")

    def __init__(self, long, type):
        self.long = long
        self.type = type


class Submodes(Enum):
    PrimeFullWindow = "ff"
    PrimeFullWindowExtended = "ef"
    PrimeLargeWindow = "lw"
    PrimeSmallWindow = "sw"
    UNDEFINED = "UND"
    #ti - timing mode
    #bu - burst mode


class Ebands(Enum):
    E1 = (1, 0.2, 0.5)
    E2 = (2, 0.5, 1.0)
    E3 = (3, 1.0, 2.0)
    E4 = (4, 2.0, 4.5)
    E5 = (5, 4.5, 12.0)

    def __init__(self, tag, min, max):
        self.tag = tag
        self.min = min
        self.max = max
