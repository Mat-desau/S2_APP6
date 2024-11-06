import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import helpers as hp

pi = np.pi

def passebas1(FreqU, FreqC, Bode, Delay, Zero, Time):
    R1 = 4750
    R2 = 4750
    R3 = 4990
    C1 = 100e-9
    C2 = 100e-9
    C3 = 22e-10
    b = []
    a = []
    tout = []
    xout = []
    yout = []

    K = R2/R1

    Wc = FreqC*2*pi
    t = np.linspace(0, 5/FreqU, 10000)
    u = np.sin(FreqU*2*pi*t)

    b, a = signal.butter(2, Wc, 'low', analog=True)

    z = np.roots(b)
    p = np.roots(a)

    tout, yout, xout = signal.lsim((b, a), u, t)

    if Bode:
        mag, ph, w, fig, ax = hp.bodeplot(b, a, 'Passe-Bas 700Hz')

    if Delay:
        hp.grpdel1(w, np.diff(ph)/np.diff(w), 'Passe-Bass 700Hz')

    if Time:
        hp.timeplt1(t, u, tout, yout, 'Passe-Bas 700Hz')

    if Zero:
        hp.pzmap1(z, p, 'Passe-Bas 700Hz')

    return b, a, z, p, K

def passebas2(FreqU, FreqC, Bode, Delay, Zero, Time):
    R15 = 2870
    R16 = 2870
    R17 = 3400
    C4 = 47e-10
    C7 = 22e-9
    C8 = 47e-10
    b = []
    a = []
    tout = []
    xout = []
    yout = []

    K = R16/R15

    Wc = FreqC*2*pi
    t = np.linspace(0, 5/FreqU, 10000)
    u = np.sin(FreqU*2*pi*t)

    b, a = signal.butter(2, Wc, 'low', analog=True)

    z = np.roots(b)
    p = np.roots(a)

    tout, yout, xout = signal.lsim((b, a), u, t)

    if Bode:
        mag, ph, w, fig, ax = hp.bodeplot(b, a, 'Passe-Bas 5000Hz')

    if Delay:
        hp.grpdel1(w, np.diff(ph)/np.diff(w), 'Passe-Bass 5000Hz')

    if Time:
        hp.timeplt1(t, u, tout, yout, 'Passe-Bas 5000Hz')

    if Zero:
        hp.pzmap1(z, p, 'Passe-Bas 5000Hz')

    return b, a, z, p, K

def passehaut1(FreqU, FreqC, Bode, Delay, Zero, Time):
    C9 = 1e-9
    C10 = 1e-9
    C11 = 1e-9
    R26 = 10700
    R27 = 48700
    b = []
    a = []
    tout = []
    xout = []
    yout = []

    K = C9/C10

    t = np.linspace(0, 5/FreqU, 10000)
    Wc = FreqC*2*pi
    u = np.sin(FreqU*2*pi*t)

    b, a = signal.butter(2, Wc, 'high', analog=True)

    z = np.roots(b)
    p = np.roots(a)

    tout, yout, xout = signal.lsim((b, a), u, t)

    if Bode:
        mag, ph, w, fig, ax = hp.bodeplot(b, a, 'Passe-Haut 1000Hz')

    if Delay:
        hp.grpdel1(w, np.diff(ph)/np.diff(w), 'Passe-Haut 1000Hz')

    if Zero:
        hp.pzmap1(z, p, 'Passe-Haut 1000Hz')

    if Time:
        hp.timeplt1(t, u, tout, yout, 'Passe-Haut 1000Hz')

    return b, a, z, p, K

def passehaut2(FreqU, FreqC, Bode, Delay, Zero, Time):
    C4 = 1e-9
    C5 = 1e-9
    C6 = 1e-9
    R12 = 7500
    R13 = 34000
    b = []
    a = []
    tout = []
    xout = []
    yout = []

    K = C4/C5

    t = np.linspace(0, 5/FreqU, 10000)
    Wc = FreqC*2*pi
    u = np.sin(FreqU*2*pi*t)

    b, a = signal.butter(2, Wc, 'high', analog=True)

    z = np.roots(b)
    p = np.roots(a)

    tout, yout, xout = signal.lsim((b, a), u, t)

    if Bode:
        mag, ph, w, fig, ax = hp.bodeplot(b, a, 'Passe-Haut 7000Hz')

    if Delay:
        hp.grpdel1(w, np.diff(ph)/np.diff(w), 'Passe-Haut 7000Hz')

    if Zero:
        hp.pzmap1(z, p, 'Passe-Haut 7000Hz')

    if Time:
        hp.timeplt1(t, u, tout, yout, 'Passe-Haut 7000Hz')

    return b, a, z, p, K

def passebande(z1, p1, k1, z2, p2, k2, FreqU, Bode, Delay, Zero, Time):
    t = np.linspace(0, 5/FreqU, 10000)
    u = np.sin(FreqU*2*pi*t)
    b = []
    a = []
    tout = []
    xout = []
    yout = []

    z, p, k = hp.seriestf(z1, p1, -k1, z2, p2, k2)
    k = -1*k
    b, a = signal.zpk2tf(z, p, k)

    tout, yout, xout = signal.lsim((b, a), u, t)

    if Bode:
        mag, ph, w, fig, ax = hp.bodeplot(b, a, 'Passe-Bande 1000Hz à 5000Hz')

    if Delay:
        hp.grpdel1(w, np.diff(ph)/np.diff(w), 'Passe-Bande 1000Hz à 5000Hz')

    if Zero:
        hp.pzmap1(z, p, 'Passe-Bande 1000Hz à 5000Hz')

    if Time:
        hp.timeplt1(t, u, tout, yout, 'Passe-Bande 1000Hz à 5000Hz')

    return b, a, z, p, k

def parallel(z1, p1, k1, z2, p2, k2, z3, p3, k3):
    zt, pt, kt = hp.paratf(z1, p1, k1, z2, p2, k2)
    kt = -1*kt
    z, p, k = hp.paratf(zt, pt, kt, z3, p3, k3)
    k = -1*k

    b, a = signal.zpk2tf(z, p, k)

    return b, a, z, p, k

def sortie(b, a, z, p, k, max, FreqU, nombre, Bode, Delay, Zero, Time):
    t = np.linspace(0, nombre/FreqU, 10000)
    u = max*np.sin(FreqU*2*pi*t)
    tout = []
    xout = []
    yout = []

    tout, yout, xout = signal.lsim((z, p, k), u, t)

    if Bode:
        mag, ph, w, fig, ax = hp.bodeplot(b, a, 'Sortie total')

    if Delay:
        hp.grpdel1(w, np.diff(ph)/np.diff(w), 'Sortie total')

    if Zero:
        hp.pzmap1(z, p, 'Sortie total')

    if Time:
        hp.timeplt1(t, u, tout, yout, 'Sortie total')

def main():
    #Graphique Bode, Graphique Delai, Graphique Zero, Graphique temps (reponse temporel)

    #, True, True, True, True
    #, False, False, False, False
    b700, a700, z700, p700, k700 = passebas1(500, 700, True, False, False, False)
    b5000, a5000, z5000, p5000, k5000 = passebas2(4000, 5000, True, False, False, False)
    b1000, a1000, z1000, p1000, k1000 = passehaut1(1200, 1000, True, False, False, False)
    b7000, a7000, z7000, p7000, k7000 = passehaut2(7500, 7000, True, False, False, False)
    bB, aB, zB, pB, kB = passebande(z1000, p1000, k1000, z5000, p5000, k5000, 2500, True, False, False, False)
    btout, atout, ztout, ptout, ktout = parallel(z700, p700, k700, zB, pB, kB, z7000, p7000, k7000)
    sortie(btout, atout, ztout, ptout, ktout, 0.25, 2500, 2, True, True, True, True)

    plt.show()

if __name__ == '__main__':
    main()