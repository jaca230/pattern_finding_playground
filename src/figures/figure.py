from abc import ABC, abstractmethod


class Figure(ABC):
    @abstractmethod
    def draw(self, *args, **kwargs):
        pass


class PlotFigure(Figure):
    def __init__(self, show: bool = True):
        self.show = show

    def _display(self, plt):
        if self.show:
            plt.show()
