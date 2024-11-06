"""
Fichiers d'exemples pour la problématique de l'APP6 (S2)
(c) JB Michaud Université de Sherbrooke
v 1.0 Hiver 2023

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp


###############################################################################
def exampleRacines():
    """
    Calcule et affiche les pôles et zéros d'une FT arbitraire

    :return:
    """
    b1 = [1, 0, 0]  # définition des facteurs du polynome numérateur
    a1 = [1, 1, 0.5]  # définition des facteurs du polynome dénominateur

    # méthode plus manuelle
    z1 = np.roots(b1)
    # pour éventuellement faire l'inverse de l'opération roots, i.e. passer d'une
    # liste de racines aux coefficients du polynôme, voir np.poly
    p1 = np.roots(a1)
    print(f'Racine exemple 1 Zéros:{z1}, Pôles:{p1}')  # affichage du resultat dans la console texte
    # appel d'une fonction pour affichage graphique, voir autre fichier helpers.py
    hp.pzmap1(z1, p1, 'Example de racines 1')

    # méthode utilisant scipy
    (z2, p2, k2) = signal.tf2zpk(b1, a1)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z2}, Pôles: {p2}, Gain: {k2}')
    hp.pzmap1(z2, p2, 'Example de racines 2')


###############################################################################
def exampleBode():
    """
    Calcule et affiche le lieu de bode d'une FT arbitraire

    :return:
    """
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 1, 0.5]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 1, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Example 1')


###############################################################################
def exampleButterworth():
    """
    Exemple de génération et affichage pour la FT d'un filtre de butterworth d'ordre 4

    :return:
    """

    order = 4
    wn = 1   # frequence de coupure = 1 rad/s
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b1, a1 = signal.butter(order, wn, 'low', analog=True)
    print(f'Butterworth Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    print(f'Racine butterworth Zéros:{np.roots(b1)}, Pôles:{np.roots(a1)}')  # affichage du resultat dans la console texte


    # Réponse en fréquence
    mag1, ph1, w1, fig, ax = hp.bodeplot(b1, a1, 'Butterworth Example')

    # Délai de groupe
    delay = - np.diff(ph1) / np.diff(w1)    # calcul
    hp.grpdel1(w1, delay, 'Exemple butterworth')    # affichage

    # Exemple 1 réponse temporelle 3 cos
    t1 = np.linspace(0, 10, 5000, endpoint=False)
    # génération de 3 sinusoîdes
    u1 = (np.cos(2 * np.pi * 0.4 * t1) + 0.6 * np.cos(2 * np.pi * 4 * t1) +
          0.5 * np.cos(2 * np.pi * 8 * t1))
    # simulation temporelle du système
    tout1, yout1, xout1 = signal.lsim((b1, a1), u1, t1)
    hp.timeplt1(t1, u1, tout1, yout1, 'filtrage de 3 sinus')   # affichage

    # Exemple 2 réponse à l'échelon à la mitaine mitaine
    t2 = np.linspace(0, 10, 5000, endpoint=False)
    u2 = np.ones_like(t2)
    tout2, yout2, xout2 = signal.lsim((b1, a1), u2, t2)
    hp.timeplt1(t2, u2, tout2, yout2, 'échelon avec lsim')

    # Exemple 3 réponse à l'échelon via scipy.signal
    tout3, yout3 = signal.step((b1, a1))
    t3m = np.amax(tout3)
    stept3 = t3m/len(tout3)
    hp.timeplt1(np.arange(0, t3m, stept3), np.ones_like(tout3),
                tout3, yout3, 'échelon avec signal.step')


###############################################################################
def probleme1():
    """
    Framework pour le problème 1 du laboratoire

    :return:
    """

    # entrez les racines de la FT #1, adaptez au besoin
    z1 = [3]
    p1 = [-3]
    k1 = 1
    b1, a1 = signal.zpk2tf(z1, p1, k1)  # obtient aussi la représentation polynomiale, une classe aurait été nice
    # ajouter le code pour générer la carte des pôles et zéros et le lieu de Bode
    # utiliser les fonctions dans helpers.py ou les exemples ci-dessus
    #hp.pzmap1(z1, p1, 'H1')
    #hp.bodeplot(b1, a1, 'H1')

    # entrez les racines de la FT #2, adaptez au besoin
    z2 = [3+5j, 3-5j]
    p2 = [-3+5j, -3-5j]
    k2 = 1
    b2, a2 = signal.zpk2tf(z2, p2, k2)
    # ajouter le code pour générer la carte des pôles et zéros et le lieu de Bode
    # utiliser les fonctions dans helpers.py ou les exemples ci-dessus
    #hp.pzmap1(z2, p2, 'H2')
    #hp.bodeplot(b2, a2, 'H2')

    # définit les entrées pour la simulation temporelle, i.e. quelques sinusoîdes de fréquence différentes
    t = np.linspace(0, 10, 5000)  # 5000 points de 0 à 10s
    w = [0, 1, 4, 15, 50]   # valeurs des fréquences désirées, rad/s
    # génère une constante si w = 0 sinon un sinus
    u = [np.ones_like(t) if w[i] == 0 else np.sin(w[i] * t) for i in range(len(w))]

    # exemple réponse temporelle pour plusieurs entrées sinusoidales définies ci-dessus
    # initialise les listes qui vont contenir l'information retournée par lsim
    tout = []
    yout1 = []
    yout2 = []
    yserie = []
    ypara = []
    # itère sur les fréquences désirées
    for i in range(len(w)):
        temp = signal.lsim((b1, a1), u[i], t)  # temp = [t, y, x], voir l'aide de lsim
        tout.append(temp[0])    # met le temps dans la liste des temps
        yout1.append(temp[1])   # met la sortie dans la liste des sorties
        temp = signal.lsim((b2, a2), u[i], t)  # répète pour l'autre FT
        yout2.append(temp[1])
        temp = signal.lsim((b2, a2), yout1[i], t)  # ici on applique successivement la sortie de H1 à l'entrée de H2
        yserie.append(temp[1])
        ypara.append(yout1[i] + yout2[i])  # pour le système en parallèle on additionne simplement les sorties
    # affichage de tout ça
    hp.timepltmulti1(t, u, w, tout, yout1, 'H1')
    hp.timepltmulti1(t, u, w, tout, yout2, 'H2')
    hp.timepltmulti1(t, u, w, tout, yserie, 'H1*H2')
    hp.timepltmulti1(t, u, w, tout, ypara, 'H1+H2')

    # autre méthode où on génère les FT série ou parallèle d'abord, au lieu de le faire à la mitaine avec les
    # signaux comme ci-dessus
    zp, pp, kp = hp.paratf(z1, p1, k1, z2, p2, k2)
    bp, ap = signal.zpk2tf(zp, pp, kp)
    print(f'H1+H2 Zéros : {zp}, Pôles: {pp}, Gain: {kp}')
    hp.pzmap1(zp, pp, 'H1+H2')
    magp, php, wp, fig, ax = hp.bodeplot(bp, ap, 'H1+H2')
    hp.grpdel1(wp, -np.diff(php)/np.diff(wp), 'H1+H2')

    zs, ps, ks = hp.seriestf(z1, p1, k1, z2, p2, k2)
    bs, a_s = signal.zpk2tf(zs, ps, ks)
    print(f'H1*H2 Zéros : {zs}, Pôles: {ps}, Gain: {ks}')
    hp.pzmap1(zs, ps, 'H1*H2')
    mags, phs, ws, fig, ax =hp.bodeplot(bs, a_s, 'H1*H2')
    hp.grpdel1(ws, -np.diff(phs)/np.diff(ws), 'H1*H2')

    # ajouter ici le code pour faire la simulation temporelle directement avec les FT série/parallèle
    # utiliser les exemples ci-dessus ou préférablement les fonctions dans helpers.py


###############################################################################
def probleme2():
    """
    Framework pour le problème 2

    :return:
    """

    # génère les filtres, ajuster au besoin
    wc = 2 * np.pi * 5000      # fréquence de coupure rad/s
    b1, a1 = signal.butter(2, wc, 'low', analog=True)
    z1, p1, k1 = signal.tf2zpk(b1, a1)
    print(f'Passe-bas Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    b2, a2 = signal.butter(2, wc, 'high', analog=True)
    z2, p2, k2 = signal.tf2zpk(b2, a2)
    print(f'Passe-haut Numérateur {b2}, Dénominateur {a2}')  # affiche les coefficients correspondants au filtre

    # génère une onde carrée, ajuster au besoin
    fsquare = 1000  # Hz
    t, step = np.linspace(0, .01, 5000, retstep=True)
    u1 = signal.square(2 * np.pi * fsquare * t, 0.5)

    # ajuste le gain de chaque bande, ajuster au besoin
    lowk = 1
    highk = 1

    # génère la combinaison parallèle
    zp, pp, kp = hp.paratf(z1, p1, k1 * lowk, z2, p2, k2 * highk)
    bp, ap = signal.zpk2tf(zp, pp, kp)
    print(f'Égaliseur klow={lowk}, khigh={highk}: Zéros : {zp}, Pôles: {pp}, Gain: {kp}')
    magp, php, wp, fig, ax = hp.bodeplot(bp, ap, f'klow={lowk}, khigh={highk}')
    hp.grpdel1(wp, -np.diff(php)/np.diff(wp), f'klow={lowk}, khigh={highk}')

    # simule tout ce beau monde
    toutp, youtp, xoutp = signal.lsim((bp, ap), u1, t)
    tout1, yout1, xout1 = signal.lsim((z1, p1, k1 * lowk), u1, t)
    tout2, yout2, xout2 = signal.lsim((z2, p2, k2 * highk), u1, t)
    yout = [yout1, yout2, youtp]
    hp.timepltmulti2(t, u1, toutp, yout, f'Égaliseur klow={lowk}, khigh={highk}', ['H1', 'H2', 'HÉgaliseur'])

    # génère une entrée sinusoïdale à la même fréquence que l'onde carrée
    fsin = fsquare
    u2 = np.sin(2*np.pi*fsin*t)

    # redéfinit les gains de bande
    lowk = .1
    highk = 1

    # ajouter le code pour regénèrer le système parallèle avec les nouveaux gains
    # ajouter le code pour resimuler tout ça
    # modifier au besoin pour l'énoncé


###############################################################################
def main():
    # décommentez la ou les lignes correspondant aux fonctions que vous voulez exécuter
    exampleRacines()
    # exampleBode()
    # exampleButterworth()
    #probleme1()
    #probleme2()
    plt.show()


#####################################
# Manière standardisée d'exécuter ce que vous désirez lorsque vous "exécutez" le fichier
# permet d'importer les fonctions définies ici comme un module dans un autre fichier .py
# voir par exemple le fichier helper.py qui n'exécutera jamais rien tout seul
# et dont les fonctions sont importées ici à titre de support pour ce qu'on essaie de faire
# pour l'instant ne rien modifier ci-dessous
if __name__ == '__main__':
    main()
