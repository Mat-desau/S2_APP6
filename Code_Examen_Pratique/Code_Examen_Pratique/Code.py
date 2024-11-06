import numpy as np
import helpers as hp
import scipy.signal as signal
import matplotlib.pyplot as plt

pi = np.pi

def dBversgain(dB):
    Gain = 10*np.exp(dB/20)
    return Gain

def gainversdB(Gain):
    dB = 20*np.log10(Gain)
    return dB

def RadtoDeg(rad):
    Deg = (rad*360)/(2*pi)
    return Deg

def DegtoRad(Deg):
    Rad = (Deg*2*pi)/360
    return Rad

def Question1a():
    w = np.linspace(0.1, 1000, 1000)
    s = 1j*w

    H1 = 100/(s+100)
    H2 = (s**2)/((s**2)+(141*s)+(100**2))

    H = H1 + H2

    G = np.abs(H)
    P = np.angle(H)
    D = -np.diff(P)/np.diff(w)

    P = RadtoDeg(P)
    D = D*1000

    Gain = plt.subplot(3, 1, 1)
    Phase = plt.subplot(3, 1, 2)
    Delai = plt.subplot(3, 1, 3)

    Gain.semilogx(w, gainversdB(G))
    Gain.set_title('Gain en dB')
    Gain.set_xlabel('Fréquence (rad/s)')
    Gain.set_ylabel('dB')
    Phase.semilogx(w, P)
    Phase.set_title('Phase en degrée')
    Phase.set_xlabel('Fréquence (rad/s)')
    Phase.set_ylabel('Phase (degrée)')
    Delai.plot(w[:-1], D)
    Delai.set_title('delai de groupe')
    Delai.set_xlabel('Frequence (rad/s)')
    Delai.set_ylabel('temps (ms)')

def Question1b():
    b1 = [100]
    a1 = [1, 100]
    b2 = [1, 0, 0]
    a2 = [1, 141, 100**2]

    z1, p1, k1 = signal.tf2zpk(b1, a1)
    z2, p2, k2 = signal.tf2zpk(b2, a2)

    t = np.linspace(0, 200/1000, 1000)
    u = 1*np.sin(100*t)

    z, p, k = hp.paratf(z1, p1, k1, z2, p2, k2)
    b, a = signal.zpk2tf(z, p, k)
    tout, yout, xout = signal.lsim((b, a), u, t)
    tout2, yout2 = signal.impulse((b, a))

    plt.figure()
    Reponse = plt.subplot(2, 1, 1)
    Reponse.plot(tout, yout)
    Reponse2 = plt.subplot(2, 1, 2)
    Reponse2.plot(tout2, yout2)

def Question2a():
    b = [1, 1, 1]
    a = [1, -1, 4]

    Z2a, P2a, K2a = signal.tf2zpk(b, a)

    print('Zeros = ', Z2a, '\nPoles = ', P2a, '\nGain = ', K2a)

    hp.pzmap1(Z2a, P2a, 'Zeros et poles du numero 2a')

def Question2b():
    b = [1, 1, 1]
    a = [1, 1, 4]

    Z2b, P2b, K2b = signal.tf2zpk(b, a)

    print('Changement de la valeur du deuxième -s à simplement s \nZeros = ', Z2b, '\nPoles = ', P2b, '\nGain = ', K2b)

    hp.pzmap1(Z2b, P2b, 'Zeros et poles du numero 2b')

def Question2c():
    b = [1, 1, 1]
    a = [1, 1, 4]

    Z2b, P2b, K2b = signal.tf2zpk(b, a)
    b, a = signal.zpk2tf(Z2b, P2b, K2b)
    w = np.linspace(10, 1000, 10000)
    w, mag, ph = signal.bode((b, a), w)
    D = -np.diff(ph)/np.diff(w)
    hp.grpdel1(w, D, 'delai')

def Question3(w):
    print('a) Calcul du numérateur et dénominatur\n')
    Z = [-2+2j, -2-2j]
    P = [-2+1j, -2-1j]
    K = 1
    print('Reussi\n')
    b, a = signal.zpk2tf(Z, P, K)
    print('b)\nDénominateur = ', a, '\nNumérateur = ', b)

    w2 = np.array([w])
    w3, Gain, Phase = signal.bode((b, a), w2)

    print('C)\nGain = ', Gain, '\nPhase = ', Phase)


def main():
    #Question1a()
    #Question1b()
    #Question2a()
    #Question2b()
    #Question2c()
    Question3(1)
    plt.show()


if __name__ == '__main__':
    main()