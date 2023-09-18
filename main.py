import numpy as np
import pandas as pd
from cleanData import clean

class modelWAM():

    def __init__(self, datos) -> None:
        self.df = datos

    def cleanData(self):
        self.df = clean(self.df)
