import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import solve


class Interpolacje:

    def __init__(self):
        self.x_fx = []
        self.y_fx = []

        self.x_w = []
        self.y_w = []
        self.param = []

        self.n = 0
        self.ILOSC = 0
        self.MODE = 0
        self.filename = ""

    def load(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        self.filename = self.filename
        data = np.loadtxt(self.filename, delimiter=",", dtype=float, skiprows=1)
        self.x_fx = data[:, [0]]
        self.y_fx = data[:, [1]]

    def interpolacja_lagrange(self, x, n):
        suma = 0

        for i in range(n):

            wynik = 1

            for j in range(n):
                if not j == i:
                    wynik *= (x - self.x_w[j]) / (self.x_w[i] - self.x_w[j])

            suma += self.y_w[i] * wynik

        return suma

    def licz_parametry(self):

        A = np.zeros([4 * (self.n - 1), 4 * (self.n - 1)])
        b = np.zeros([4 * (self.n - 1), 1])

        ########### 1 #############
        # 1 * a_0 = f(x_0)
        for j in range(self.n - 1):
            wiersz = np.zeros([1, 4 * (self.n - 1)])
            wiersz[0][4 * j] = 1
            A[4 * j] = wiersz
            b[4 * j] = float(self.y_w[j])

        ########### 2 #############
        # a_0 + b_0 * h + c_0 * h**2 + d_0 * h**3 = f(x_0)
        for j in range(self.n - 1):
            h = float(self.x_w[j + 1]) - float(self.x_w[j])
            y = self.y_w[j + 1]
            wiersz = np.zeros([1, 4 * (self.n - 1)])

            wiersz[0][4 * j] = 1
            wiersz[0][4 * j + 1] = h
            wiersz[0][4 * j + 2] = h ** 2
            wiersz[0][4 * j + 3] = h ** 3

            A[4 * j + 1] = wiersz
            b[4 * j + 1] = float(y)

        ########### 3 #############
        # S_j-1'(x_j) = S_j'(x_j) po przeksztalceniu
        for j in range(0, self.n - 2):
            wiersz = np.zeros([1, 4 * (self.n - 1)])
            h = float(self.x_w[j + 1]) - float(self.x_w[j])
            y = self.y_w[j + 1]

            wiersz[0][4 * j + 1] = 1  # b0
            wiersz[0][4 * j + 2] = 2 * h  # c0
            wiersz[0][4 * j + 3] = 3 * h ** 2  # d0
            wiersz[0][4 * j + 5] = -1  # b1

            A[4 * j + 2] = wiersz
            b[4 * j + 2] = float(0)

        ########### 4 #############
        # S_j-1''(x_j) = S_j''(x_j) po przeksztalceniu
        for j in range(0, self.n - 2):
            wiersz = np.zeros([1, 4 * (self.n - 1)])
            h = float(self.x_w[j + 1]) - float(self.x_w[j])
            y = self.y_w[j + 1]

            wiersz[0][4 * j + 2] = 2
            wiersz[0][4 * j + 3] = 6 * h
            wiersz[0][4 * j + 6] = -2

            A[4 * j + 3] = wiersz
            b[4 * j + 3] = float(0)

        ########### 5 #############
        # krawedz lewa
        # S_0''(x_0) = 0
        wiersz = np.zeros([1, 4 * (self.n - 1)])
        wiersz[0][2] = 2

        A[4 * (self.n - 2) + 2] = wiersz
        b[4 * (self.n - 2) + 2] = float(0)

        # krawedz prawa
        # S_0''(x_0) = 0
        h = float(self.x_w[-1]) - float(self.x_w[-2])
        wiersz = np.zeros([1, 4 * (self.n - 1)])
        wiersz[0][4 * (self.n - 2) + 2] = 2
        wiersz[0][4 * (self.n - 2) + 3] = 6 * h

        A[4 * (self.n - 2) + 3] = wiersz
        b[4 * (self.n - 2) + 3] = float(0)

        wynik = np.linalg.solve(A, b)
        return wynik

    def interpolacja_splajnami(self, x):
        i = 0
        while True:
            if x >= self.x_w[i] and x <= self.x_w[i + 1]:
                break
            else:
                i += 1

        a = self.param[4 * i]
        b = self.param[4 * i + 1]
        c = self.param[4 * i + 2]
        d = self.param[4 * i + 3]

        h = float(x) - float(self.x_w[i])
        return a + b * h + c * h ** 2 + d * h ** 3

    def zadanie1(self):
        plt.figure(figsize=(10, 6))

        self.load()
        a1 = self.ILOSC / 4

        if self.MODE == 0:
            idx = np.round(np.linspace(0, len(self.x_fx) - 1, self.ILOSC)).astype(int)
        elif self.MODE == 1:
            app1 = np.round(np.linspace(0, (len(self.x_fx) - 1) / 3, int(round(a1, 0)))).astype(int)
            app1 = np.delete(app1, -1)
            app2 = np.round(np.linspace((len(self.x_fx) - 1) / 3 + 1, (len(self.x_fx) - 1) * 2 / 3, int(round(2 * a1, 0)))).astype(int)
            app2 = np.delete(app2, -1)
            app3 = np.round(np.linspace((len(self.x_fx) - 1) * 2 / 3 + 1, (len(self.x_fx) - 1), int(round(a1, 0)))).astype(int)
            idx = np.concatenate([app1, app2, app3])
        else:
            app1 = np.round(np.linspace(0, (len(self.x_fx) - 1) / 3, int(round(2 * a1, 0)))).astype(int)
            app1 = np.delete(app1, -1)
            app2 = np.round(np.linspace((len(self.x_fx) - 1) / 3 + 1, (len(self.x_fx) - 1) * 2 / 3, int(round(a1, 0)))).astype(int)
            app2 = np.delete(app2, -1)
            app3 = np.round(np.linspace((len(self.x_fx) - 1) * 2 / 3 + 1, (len(self.x_fx) - 1), int(round(2 * a1, 0)))).astype(int)
            idx = np.concatenate([app1, app2, app3])

        self.x_w = self.x_fx[idx]
        self.y_w = self.y_fx[idx]
        n = self.x_w.shape[0]

        x = np.linspace(self.x_w[0], self.x_w[-1], 1000)
        plt.plot(self.x_fx, self.y_fx, color='green', label='wartosci prawdziwe')
        plt.scatter(self.x_w, self.y_w, color='orange', label='wezly', marker="o")
        plt.plot(x, self.interpolacja_lagrange(x, n), color='blue', label='wartosci interpolowane metodą Lagrangea\'a')
        plt.title(f"{self.filename}, interpolowane metodą Lagrange dla {self.ILOSC} węzłów")
        plt.legend()
        plt.ylim([(min(self.y_fx) - 10), (max(self.y_fx) + 10)])
        plt.show()

    def zadanie2(self):
        plt.figure(figsize=(10, 6))

        self.load()
        a1 = self.ILOSC / 4

        if self.MODE == 0:
            idx = np.round(np.linspace(0, len(self.x_fx) - 1, self.ILOSC)).astype(int)
        elif self.MODE == 1:
            app1 = np.round(np.linspace(0, (len(self.x_fx) - 1) / 3, int(round(a1, 0)))).astype(int)
            app1 = np.delete(app1, -1)
            app2 = np.round(np.linspace((len(self.x_fx) - 1) / 3 + 1, (len(self.x_fx) - 1) * 2 / 3, int(round(2 * a1, 0)))).astype(int)
            app2 = np.delete(app2, -1)
            app3 = np.round(np.linspace((len(self.x_fx) - 1) * 2 / 3 + 1, (len(self.x_fx) - 1), int(round(a1, 0)))).astype(int)
            idx = np.concatenate([app1, app2, app3])
        else:
            app1 = np.round(np.linspace(0, (len(self.x_fx) - 1) / 3, int(round(2 * a1, 0)))).astype(int)
            app1 = np.delete(app1, -1)
            app2 = np.round(np.linspace((len(self.x_fx) - 1) / 3 + 1, (len(self.x_fx) - 1) * 2 / 3, int(round(a1, 0)))).astype(int)
            app2 = np.delete(app2, -1)
            app3 = np.round(np.linspace((len(self.x_fx) - 1) * 2 / 3 + 1, (len(self.x_fx) - 1), int(round(2 * a1, 0)))).astype(int)
            idx = np.concatenate([app1, app2, app3])

        self.x_w = self.x_fx[idx]
        self.y_w = self.y_fx[idx]
        self.n = self.x_w.shape[0]
        self.param = self.licz_parametry()

        x = np.linspace(self.x_w[0], self.x_w[-1], 10000)
        y = np.zeros([1, 10000])

        for i, xi in enumerate(x):
            y[0, i] = self.interpolacja_splajnami(xi)

        y = y.T
        plt.plot(self.x_fx, self.y_fx, color='green', label='wartosci prawdziwe')
        plt.scatter(self.x_w, self.y_w, color='orange', label='wezly', marker="o")
        plt.plot(x, y, color='blue', label='wartosci interpolowane splajnami 3 st.')
        plt.title(f"{self.filename}, interpolowane metodą Splajnów 3 stopnia dla {self.ILOSC} węzłów")
        plt.legend()
        plt.show()


if __name__ == '__main__':

    WIECEJ_NA_BRZEGACH = 2
    WIECEJ_NA_SRODKU = 1
    ROWNOMIERNY = 0

    i1 = Interpolacje()

    i1.ILOSC = 20

    i1.MODE = ROWNOMIERNY
    i1.filename = "SpacerniakGdansk.csv"
    i1.zadanie1()
    i1.zadanie2()

    i1.filename = "Obiadek.csv"
    i1.zadanie1()
    i1.zadanie2()

    i1.filename = "MountEverest.csv"
    i1.zadanie1()
    i1.zadanie2()

    i1.filename = "WielkiKanionKolorado.csv"
    i1.zadanie1()
    i1.zadanie2()


