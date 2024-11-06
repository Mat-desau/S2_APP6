import numpy as np
import helpers as hp
import matplotlib.pyplot as plt
import scipy.signal as signal

pi = np.pi

def Gain2dB(Gain):
    dB = 20*np.log10(Gain)
    return dB

def Rad2deg(Rad):
    Deg = (Rad*360)/(2*pi)
    return Deg

def Deg2rad(Deg):
    Rad = (Deg*2*pi)/(360)
    return Rad

def Question1():
    print('1)\na) Calcul et affichage des lieu de bodes\n')
    w = np.linspace(0.1, 1000, 1000)
    s = 1j*w
    H1 = 100/(s+100)
    H2 = (s**2)/((s**2)+(141*s)+(100**2))
    H = H1 + H2

    Phase = np.angle(H)
    Gain = np.abs(H)
    Delai = -np.diff(Phase)/np.diff(w)

    Gain = Gain2dB(Gain)
    Phase = Rad2deg(Phase)
    Delai = Delai*1000

    plt.figure()
    Gphase = plt.subplot(3, 1, 1)
    Ggain = plt.subplot(3, 1, 2)
    Gdelai = plt.subplot(3, 1, 3)

    Gphase.semilogx(w, Phase)
    Gphase.set_title('Phase')
    Gphase.set_xlabel('Fréquence (rad/s)')
    Gphase.set_ylabel('Phase (deg)')
    Ggain.semilogx(w, Gain)
    Ggain.set_title('Gain')
    Ggain.set_xlabel('Fréquence (rad/s)')
    Ggain.set_ylabel('Gain (dB)')
    Gdelai.plot(Delai)
    Gdelai.set_title('Delai')
    Gdelai.set_xlabel('Fréquence (rad/s)')
    Gdelai.set_ylabel('Temps (ms)')

    print('b)Reponses impultionelle\n')
    t = np.linspace(0, 200/1000, 1000)
    w = 100
    u = 1*np.sin(w*t)

    b1 = [100]
    a1 = [1, 100]
    b2 = [1, 0, 0]
    a2 = [1, 141, 100**2]

    z1, p1, k1 = signal.tf2zpk(b1, a1)
    z2, p2, k2 = signal.tf2zpk(b2, a2)

    z, p, k = hp.paratf(z1, p1, k1, z2, p2, k2)

    Si = plt.subplot(2, 1, 1)
    Im = plt.subplot(2, 1, 2)
    b, a = signal.zpk2tf(z, p, k)
    tout, yout, xout = signal.lsim((b,a), u, t)
    Si.plot(tout, yout)
    tout2, yout2 = signal.impulse((b, a))
    Im.plot(tout2, yout2)

def Question2():
    print('2)\na) Calcul des pôles et des zeros\n')
    b = [1, 1, 1]
    a = [1, -1, 4]

    z, p, k = signal.tf2zpk(b, a)
    print('Zeros = ', z, '\nPoles = ', p, '\n')
    hp.pzmap1(z, p, 'Numéro 2a)')

    print('b) Déplacement du pôle pour stabilité\n')
    print('Dénominateur avant = ', a)
    a = [1, 1, 4]
    print('\nNouveau dénominateur = ', a, '\n')
    z, p, k = signal.tf2zpk(b, a)
    print('Zeros = ', z, '\nPoles = ', p, '\n')
    hp.pzmap1(z, p, 'Numéro 2b)')

    w = np.linspace(10, 1000, 10000)
    w, mag, ph = signal.bode((b, a), w)
    ph = Deg2rad(ph)
    Delai = -np.diff(ph)/np.diff(w)
    hp.grpdel1(w, Delai, 'Numéro 2c)')

def Question3(w):
    print('3)\na)Calculer le numérateur et Dénominateur\n')
    z = [-2+2j, -2-2j]
    p = [-2+1j, -2-1j]
    k = 1
    b, a = signal.zpk2tf(z, p, k)
    print('Calcul fonctionner\n')

    print('b)Affichage des polynomes\n')
    print('Dénominateur = ', a, '\nNumérateur = ', b, '\n')

    print('c) Calcul du gain et de la phase\n')
    w2 = np.array([w])
    w, mag, ph = signal.bode((b, a), w)

    print('Gain = ', mag, 'V/V\nPhase = ', ph, 'rad\n')

def main():
    Question1()
    Question2()
    Question3(1)
    plt.show()
    return 0

if __name__ == '__main__':
    main()