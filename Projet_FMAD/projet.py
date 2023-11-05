### PROJET ECO SCORE


# INSTALL
pip install numpy
pip install matplotlib
pip install numpy
pip install mip

# IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mip import Model, xsum, minimize, BINARY, CONTINUOUS
from math import log, pow


## QUESTION 1
exc = pd.ExcelFile("AGRIBALYSE3.1_partie agriculture_conv_vf.xlsx")
df = pd.read_excel(exc, 'AGB_agri_conv', usecols = [0]+[i for i in range(3,20)])
data = df[2:].values
donnees = []
for i in data :
    donnees.append(i[2:])


## QUESTION 2
def normaliser(donnees):
    facteurs = [7550.0, 0.0523, 4220.0, 40.9, 0.000595, 0.000129, 0.0000173, 55.6, 1.61, 19.5, 177.0, 56700.0, 819000.0, 11500.0, 65000.0, 0.0636]
    for i in donnees:
        for j in range(0, 16):
            i[j] = i[j] /facteurs[j]
    return donnees

donnees = normaliser(donnees)


## QUESTION 3
def graphiques(donnees) :
    critere1 = []
    critere14 = []
    critere15 = []
    for i in donnees :
        critere1.append(i[0])
        critere14.append(i[13])
        critere15.append(i[15])
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].scatter(critere1, critere14, s = 10)
    axs[0, 0].set_title("Critère 14 en fonction du critère 1")
    axs[1, 0].scatter(critere1, critere15, s = 10)
    axs[1, 0].set_title("Critère 15 en fonction du critère 1")
    axs[0, 1].scatter(critere14, critere1, s = 10)
    axs[0, 1].set_title("Critère 1 en fonction du critère 14")
    axs[1, 1].scatter(critere14, critere15, s = 10)
    axs[1, 1].set_title("Critère 15 en fonction du critère 14")
    axs[0, 2].scatter(critere15, critere1, s = 10)
    axs[0, 2].set_title("Critère 1 en fonction du critère 15")
    axs[1, 2].scatter(critere15, critere14, s = 10)
    axs[1, 2].set_title("Critère 14 en fonction du critère 15")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()

graphiques(donnees)


## QUESTION 5
def pareto_domine(X, Y): # X et Y sont les listes (xi)i et (yi)i

  # on veut xi >= yi pour tout i
  for i in range(len(X)):
    if X[i]<Y[i]:
      return False
  # on veut xi > yi pour au moins un i
  for i in range(len(X)):
    if X[i]> Y[i]:
      return True

  return False

# TEST:
# Définition de deux ensembles de solutions
C = [2, 3, 4, 5, 7, 7]
D = [2, 3, 4, 5, 6, 7]

# Vérification si les relations C pareto-Domine D
if pareto_domine(C,D):
  print("C pareto domine D.")
else:
  print("C ne pareto domine pas B.")


## QUESTION 6
def est_transitive(R):
  for x, y in R:
    for z, w in R:
      if y == z and (x, w) not in R:
        return False
  return True

def est_irreflexive(R):
  for x, y in R: # = (x,y)
    if x == y:
      return False
  return True

def est_asymetrique(R):
  for x, y in R:
    if (y, x) in R:
      return False
  return True


## QUESTION 7
def pareto_2(x,y):
    l = [0, 13, 15]
    inf = 0
    for i in l :
        if (x[i] < y[i]) :
            inf+=1
        if (x[i] > y[i]) :
            return (False)
    if (inf != 0) :
        return (True)
    return (False)

def pourcentage(donnees) :
    res_1 = 0
    res_2 = 0
    for i in range(len(donnees)) :
        for j in range(len(donnees)) :
            if (pareto_domine(donnees[i], donnees[j]) and i != j) :
                res_1+=1
            if (pareto_2(donnees[i], donnees[j]) and i != j) :
                res_2+=1
    print("Pourcentage pour tous les critères : ", round(100 * res_1 / ((len(donnees) - 1)*len(donnees)), 2), "%")
    print("Pourcentage pour les critères 1, 14 et 16: ", round(100 * res_2 / ((len(donnees) - 1)*len(donnees)), 2), "%")


## QUESTION 10
def L1_inv(X, Y, wo):
  m = Model()
  n = 16
  w = [ m.add_var(lb=0, ub=1) for i in range(n) ]
  d = [ m.add_var(lb=0) for i in range(n) ]

  m.add_constr(sum(w) == 1)
  m.add_constr(sum(w[i] * X[i] for i in range(n)) - sum(w[i] * Y[i] for i in range(n)) >= 0)

  for i in range(16):
        m.add_constr(w[i] - wo[i] <= d[i])  # d[i] >= w[i] - wo[i]
        m.add_constr(wo[i] - w[i] <= d[i])  # d[i] >= wo[i] - w[i]

  m.objective = minimize(xsum(d[i] for i in range(n)))
  m.optimize()

  return m.objective_value


## QUESTION 11
# Définition de la fonction count
def count(donnees, x):
  res = 0
  l = []
  for i in donnees:
    for j in donnees:
      val = L1_inv(i, j, wo)
      if val is not None:
        l.append(val)
  for k in l:
    if k <= x:
      res += 1
  return res

def graphique_L1(donnees):
    # Calcul des valeurs x et y pour le tracé du graphique
    x = np.arange(0, 2, 0.05)
    y = [count(donnees, xi) for xi in x]

    # Tracé du graphique
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("Nombre de paires (X, Y) telles que L1_inv(X, Y) <= x")
    plt.title("Graphique associant à chaque valeur x, le nombre de paires (X, Y) telles que L1_inv(X, Y) <= x")
    plt.show()

graphique_L1(donnees)


## QUESTION 12
def moyenne_ponderee(X, w):
  n= len(X)
  permute = np.argsort(X)  # x1 <= xn
  Xp=[0]*n
  for i in range(n):
    Xp[i]= X[permute[i]]
  moy =0
  for i in range(n):
    moy+= w[i]* Xp[i]

  return(moy)

def racine(x,i):
  return x ** (1/i)

def moyenne_ponderee_geo(X, w):
  n= len(X)
  sommep= 1
  m=0
  for x in w:
    m+= x
  for i in range(n):
    sommep *= pow(X[i], w[i])
  return racine(sommep, m)


## QUESTION 15
def R1(x, y) :
    for i in range(len(x)) :
        if (x[i] >= y[i]) : #Inférieur strict
            return (False)
    return (True)

def R2(x, y) :
    for i in range(len(x)) :
        if (x[i] <= y[i]) : #Supérieur strict
            return (False)
    return (True)

def kendall(data, R1, R2, p1, p2) :
    res = 0
    if (p1 + p2 != 1):
        return (None)
    for i in data :
        for j in data :
            if (R1(i, j) and R2(j, i)) :
                res += p1
            if ((R1(i, j) and not R2(i, j)) or (not R1(j, i) and R2(j, i))) :
                res+=p2
    return (res)


## QUESTION 17
wo = [0.2106, 0.0631, 0.0501, 0.0478, 0.0896, 0.0184, 0.0213, 0.062, 0.028, 0.0298, 0.0371, 0.0192, 0.0794, 0.0851, 0.0832, 0.0755]
w1 = [0.05, 0.07, 0.03, 0.08, 0.1, 0.04, 0.11, 0.02, 0.09, 0.06, 0.01, 0.12, 0.02, 0.1, 0.05, 0.05]
w2 = [1/16]*16

# Fonction de catégorisation d'une alternative en fonction de son score
def categorisation(score):
    if score > 80:
        return "A"
    elif score > 60:
        return "B"
    elif score > 40:
        return "C"
    elif score > 20:
        return "D"
    else:
        return "E"

def sup_cat(x): #retourne la borne inférieure de la catégorie supérieure pour un x donné
    cat = categorisation(x)
    if cat == "A":
        return 100
    elif cat == "B":
        return 81
    elif cat == "C":
        return 61
    elif cat == "D":
        return 41
    else:
        return 21


def inf_cat(x): #retourne la borne supérieure de la catégorie inférieure pour un x donné
    cat = categorisation(x)
    if cat == "A":
        return 80
    elif cat == "B":
        return 60
    elif cat == "C":
        return 40
    elif cat == "D":
        return 20
    else:
        return 0



def L1_inv_eco(X, Y, wo):
    m = Model()
    n = 16
    w = [ m.add_var(lb=0, ub=1) for i in range(n) ]
    d = [ m.add_var(lb=0) for i in range(n) ]

    m.add_constr(sum(w) == 1)

    s = score(xsum(X[i]*w[i] for i in range(n)))
    m.add_var(var_type=BINARY)
    y = m.vars[-1]

    if s > sup_cat(x):
        m += y == 1
    else:
        m += y == 0

  m.add_constr(sum(w[i] * X[i] for i in range(n))  <= sup_cat(somme_pond(X, wo)) - 10000*y  )
  m.add_constr(sum(w[i] * X[i] for i in range(n))  >= inf_cat(somme_pond(X, wo)) + 10000*(1-y ))

  for i in range(16):
        m.add_constr(w[i] - wo[i] <= d[i])  # d[i] >= w[i] - wo[i]
        m.add_constr(wo[i] - w[i] <= d[i])  # d[i] >= wo[i] - w[i]

  m.objective = minimize(xsum(d[i] for i in range(n)))
  m.optimize()

  return m.objective_value

# Définition de la fonction count
def count_eco(donnees, x):
  res = 0
  l = []
  for i in donnees:
    for j in donnees:
      val = L1_inv_eco(i, j, wo)
      if val is not None:
        l.append(val)
  for k in l:
    if k <= x:
      res += 1
  return res

def graphique_L1_eco(donnees):
    # Calcul des valeurs x et y pour le tracé du graphique
    x = np.arange(0, 2, 0.05)
    y = [count(donnees, xi) for xi in x]

    # Tracé du graphique
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("Nombre de paires (X, Y) telles que L1_inv(X, Y) <= x")
    plt.title("Graphique associant à chaque valeur x, le nombre de paires (X, Y) telles que L1_inv(X, Y) <= x")
    plt.show()

graphique_L1_eco(donnees)


## QUESTION 17 BIS

#somme ponderee
def somme_pond(X, w):
  sum = 0
  for i in range(16):
    sum+= X[i]*w[i]
  return sum

# Fonction de calcul du score eco
def score(x):
  s = 100 - ( (20* (log(10*x + 1))) / (log(2 + (1/(100*(x**4)) ) ) ) )
  return(s)

# Fonction de catégorisation d'une alternative en fonction de son score
def categorisation(score):
    if score > 80:
        return "A"
    elif score > 60:
        return "B"
    elif score > 40:
        return "C"
    elif score > 20:
        return "D"
    else:
        return "E"

def compte(donnees, wo, w):
  cpt = 0
  for d in donnees:
    cat_wo = categorisation( score( somme_pond(d, wo)))
    cat_w = categorisation(score( somme_pond(d, w)))
    if cat_wo != cat_w:
      cpt += 1
  return "Sur les " + str(len(donnees)) +" alternatives, il y en a " + str(cpt)+ " qui ont changé de categorie, losque on change le poids w des criteres ."

compte(donnees, wo, w1)










