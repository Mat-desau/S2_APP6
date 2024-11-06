import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import helpers as hp

pi = np.pi

def passe_bas (fc, fu, k, wn):
    wr = 2*pi*wn
    t = np.linspace(0, 5/fu, 1000)
    wc = 2 * pi * fc
    u = np.sin(2*pi*fu*t)

    b, a = signal.butter(2, wc, 'low', analog=True)

    b = b*k

    tout, yout, xout = signal.lsim((b, a), u, t)

    #mag, ph, w, f, fig, ax = hp.bodeplot(b, a, wr, 'passe-bas')

    #hp.grpdel1(f, -np.diff(ph)/np.diff(w), 'passe-bas')

    #hp.timeplt1(t, u, tout, yout, 'passe-bas')

    return b, a

def passe_haut (fc, fu, k, wn):
    wr = 2 * pi * wn
    t = np.linspace(0, 5/fu, 1000)
    wc = 2 * pi * fc
    u = np.sin(2*pi*fu*t)

    b, a = signal.butter(2, wc, 'high', analog=True)

    z = np.roots(b)
    p = np.roots(a)

    b = b*k

    tout, yout, xout = signal.lsim((b, a), u, t)

    mag, ph, w, f, fig, ax = hp.bodeplot(b, a, wr, 'passe-haut')

    hp.grpdel1(f, -np.diff(ph)/np.diff(w), 'passe-haut')

    hp.pzmap1(z, p, 'Passe-haut')

    #hp.timeplt1(t, u, tout, yout, 'passe-haut')

    return b, a

def passe_bande(fch, kh, fcb, kb, fu, wn):
    bh, ah = passe_haut(fch, fu, kh, wn)
    bb, ab = passe_bas(fcb, fu, kb, wn)

    wr = 2 * pi * wn
    t = np.linspace(0, 5 / fu, 1000)
    u = 0.25*np.sin(2 * pi * fu * t)

    b, a = hp. seriestf(bh, ah, bb, ab)

    tout, yout, xout = signal.lsim((b, a), u, t)

    #mag, ph, w, f, fig, ax = hp.bodeplot(b, a, wr, 'passe-bande')

    #hp.grpdel1(f, -np.diff(ph) / np.diff(w), 'passe-bande')

    #hp.timeplt1(t, u, tout, yout, 'passe-bande')

    return b, a

def tout_le_kit(k1 , k2, fu, wn):
    wr = 2 * pi * wn
    t = np.linspace(0, 5 / 2500, 1000)
    u = [0.25*np.sin(2 * pi * fu[i] * t) for i in range(len(fu))]

    b1, a1 = passe_bas(700, 500, -1, wn)
    b1 = b1 * k1

    b2, a2 = passe_haut(7000, 10000, -1, wn)

    b3, a3 = passe_bande(1000, -1, 5000, -1, 3000, wn)
    b3 = b3 * k2

    bt, at = hp.paratf(b1, a1, b2, a2)

    b, a = hp.paratf(bt, at, b3, a3)

    tout = []
    yout = []

    for i in range(len(u)):
        toutt, youtt, xoutt = signal.lsim((b, a), u[i], t)
        tout.append(toutt)
        yout.append(youtt)

    mag, ph, w, f, fig, ax = hp.bodeplot(b, a, wr, 'Circuit complet')

    hp.grpdel1(f, -np.diff(ph) / np.diff(w), 'Circuit complet')

    #hp.timepltmulti1(t, u, fu, tout, yout, 'tout')

    return b, a

def main():
    wn = np.linspace(1, 100000, 100000)
    fu = np.array([20, 200, 2500, 20000])

    #b, a = passe_bas(700, 500, -1, wn)

    #b, a = passe_haut(7000, 10000, -1 , wn)

    #b, a = passe_bande(1000, -1, 5000, -1, 2500, wn)

    b, a = tout_le_kit(1, (47500/61000), fu, wn)

    print('b:',b, '\n','a: ', a)

    plt.show()

if __name__ == '__main__':
    main()

