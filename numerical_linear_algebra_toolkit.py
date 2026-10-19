import numpy as np
from numpy import array
from numpy import arange
from numpy import zeros
from numpy import ones
from numpy import max
from numpy import argmax
from numpy import min
from numpy import argmin
from numpy import abs
from numpy import absolute
from numpy import sqrt
from numpy import sort
from numpy import argsort
from numpy import dot
from numpy import eye
from numpy import copy
from numpy import diag
from numpy import tril 
from numpy import triu
from numpy import delete
from numpy import setdiff1d
from numpy import inf
from numpy import float64
from numpy import set_printoptions
from numpy import printoptions
from numpy import diag_indices
from numpy import unravel_index
from numpy.linalg import norm
from numpy.linalg import trace
from numpy.linalg import det

from numpy.linalg import eigvals # La funció eigvals es pot fer servir a la
                                 # funció radi_espectral i per comprovar
                                 # les funcions que fan el càlcul dels VEPS
# 
# Mètodes directes
#

# Solució de sistemes triangulars

def sol_sti(A, b, uns_a_la_diagonal = True):

    n = len(A)
    for i in range(n):
        for j in range(i):
            b[i] -= A[i, j]*b[j]
        if not uns_a_la_diagonal:
            b[i] /= A[i, i]
    return b

def sol_sts(A, b, uns_a_la_diagonal = False):

    n = len(A)
    for i in range(n-1, -1, -1):
        for j in range (i+1, n):
            b[i] -= A[i, j]*b[j]
        if not uns_a_la_diagonal:
            b[i] /= A[i, i]
    return b

# Factorització LU: LU sense pivotatge, amb pivotatge parcial i pivotatge
# parcial esglaonat 

def lu(A):

    A = A.copy()
    for k in range (len(A)):
        for i in range (k+1, len(A)):
            m = A[i][k]/A[k][k]
            for j in range (k+1, len (A)):
                A[i][j]= A[i][j] - m*A[k][j]
            A[i][k] = m
    return A 

def lu_pp(A):

    A = A.copy()
    n = len(A)
    p = np.arange(n) #aquest es el vector de permutacio
    for k in range (len(A)):
        max = np.abs(A[k, k])
        pos_max = k
        for fila in range(k+1, n):
            if np.abs(A[fila, k]) > max:
                pos_max = fila
                max = np.abs(A[fila, k])
        A[[k, pos_max]] = A[[pos_max, k]]
        p[[k, pos_max]] = p[[pos_max, k]]
        for i in range (k+1, len(A)):
            m = A[i][k]/A[k][k]
            for j in range (k+1, len (A)):
                A[i][j]= A[i][j] - m*A[k][j]
            A[i][k] = m
    return A, p 

def lu_ppe(A):

    A = A.copy()
    n = len(A)
    p = np.arange(n) #aquest es el vector de permutacio
    for k in range (len(A)):
        max_ratio = np.abs(A[k, k])/np.linalg.norm(A[k, k:], ord=np.inf)
        pos_max = k
        for fila in range(k+1, n):
            temp = np.abs(A[fila, k])/np.linalg.norm(A[fila, k:], ord=np.inf) #ho guardem com a valor temporal per no calcular-lo 2 vegades
            if temp > max_ratio:
                pos_max = fila
                max_ratio = temp
        A[[k, pos_max]] = A[[pos_max, k]]
        p[[k, pos_max]] = p[[pos_max, k]]
        for i in range (k+1, len(A)):
            m = A[i][k]/A[k][k]
            for j in range (k+1, len (A)):
                A[i][j]= A[i][j] - m*A[k][j]
            A[i][k] = m
    return A, p 
# Esquemes compactes: Doolittle, Crout, LDLT i Choleski

def doolittle(A):

    A = A.copy()
    n = len(A)
    for i in range (1, n):
        A[i, 0] = A[i, 0]/A[0, 0]
    for k in range (1, n):
        for j in range (k, n):
            for p in range (k):
                A[k, j] = A[k, j] - A[k, p]*A[p, j]
        for i in range (k+1, n):
            s = 0
            for p in range (k):
                s += A[i, p]*A[p, k]
            A[i, k] = (A[i, k] - s)/A[k, k]
    return A

def crout(A):

    A = A.copy()
    n = len(A)
    for j in range (1, n):
        A[0, j] = A[0, j]/A[0, 0]
    for k in range (1, n):
        for i in range (k, n):
            for p in range (k):
                A[i, k] -= A[i, p]*A[p, k]
        for j in range (k+1, n):
            for p in range (k):
                A[k, j] -= A[k, p]*A[p, j]
            A[k, j] /= A[k, k]
    return A

def ldlt(A):

    A = A.copy()
    n = len(A)
    for i in range(1, n):
        A[i, 0] /= A[0, 0]
    for k in range(1, n):
        for r in range(k):
            A[k, k] -= A[k, r]**2*A[r, r]
        for i in range (k+1, n):
            for r in range (k):
                A[i, k] -= A[i, r]*A[r, r]*A[k, r]
            A[i, k] /= A[k, k]
    return A

def choleski(A):

    A = A.copy()
    n = len(A)
    A[0, 0] = sqrt(A[0, 0])
    for i in range (1, n):
        A[i, 0] /= A[0, 0]
    for j in range (1, n):
        for k in range (j):
            A[j, j] -= A[j, k]**2
        A[j, j] = sqrt(A[j, j])
        for i in range(j+1, n):
            for k in range(j):
                A[i, j] -= A[i, k]*A[j, k]
            A[i, j] /= A[j, j]
    return A

# Solució a partir de la factorització LU de la matriu A del sistema lineal
# Ax = b

def sol_lu(A, b, l_amb_uns_a_la_diagonal = True):

    #Primer hem de resoldre Ly = b (per trobar y) i després Ux = y
    n = len(A)
    if l_amb_uns_a_la_diagonal:
        U = np.triu(A)
        L = A - U + np.eye (n)
        y = sol_sti (L, b, True)
        b = sol_sts (U, y, False)
    else: #Llavors U es la que te uns a la diagonal
        L = np.tril(A)
        U = A - L + np.eye(n)
        y = sol_sti(L, b, False)
        b = sol_sts(U, y, True)
    return b

# Factorització QR: Mètodes de Gram-Schmidt classic, Gram-Schmidt modificat i
# Householder 

def gs_classic(a):

    a = a.copy()
    n = len(a[0]) #(nombre de columnes)
    m = len(a)
    r = np.zeros((n, n))
    r[0, 0] = np.linalg.norm(a[:, 0])
    a[:, 0] /= r[0, 0]
    for k in range (1, n):
        for j in range (k):
            for p in range (m):
                r[j, k] += a[p, j]*a[p, k]
        for p in range (k):
            a[:, k] -= r[p, k]*a[:, p]
        r[k, k] = np.linalg.norm(a[:, k])
        a[:, k] /= r[k, k]
    return a, r

def gs_modificat(a):

    a = a.copy()
    n = len(a[0])
    m = len(a)
    r = np.zeros((n,n))
    r[0, 0] = np.linalg.norm(a[:, 0])
    a[:, 0] /= r[0, 0]
    for k in range (1, n):
        for s in range (k , n):
            for p in range (m):
                r[k-1, s] += a[p, k-1]*a[p, s]
            a[:, s] -= r[k-1, s]*a[:, k-1]
        r[k, k] = np.linalg.norm(a[:, k])
        a[:, k] /= r[k, k]
    return a, r

def house(x, tol=1.0e-14):

    n = len(x)
    u = np.zeros(n)
    if np.linalg.norm(x[1:]) < tol:
        beta = 0
        u[0] = 1
        """retornem el vector e(1), pero dona igual perque com beta es 0 la matriu
        De householder sera la identitat, independentment del vector u asignat en aquest cas.
        Es a dir, si la norma es menor que la tolerancia no fem cap transformació
        """
    else:
        v = np.linalg.norm(x)
        u = np.copy(x)
        if x[0] < 0:
            u[0] -= v
        else:
            u[0] += v
        beta = 1/(v*np.abs(u[0]))
    return beta, u

def qr_house(a):

    a = a.copy()
    n = len(a[0])
    m = len(a)
    Qt = np.eye(m) #hem d'aplicar les transformacions de Householder a la matriu identitat, la qual es transformara en Qt
    #no calcularem les matrius de householder, aplicarem una propietat (la propietat 8 en el manuscrit que ens permet obtenir els vectors sente crear la matri propia)
    for k in range(n-1):
        beta, u = house(a[k:, k])
        for j in range(k, n):
            utx = 0
            for p in range(len(u)):
                utx += u[p]*a[k+p, j]
            a[k:, j] -= beta*utx*u
        for i in range (m):
            utx = 0
            for p in range(len(u)):
                utx += u[p]*Qt[k+p, i]
            Qt[k:, i] -= beta*utx*u
    if m != n:
        beta, u = house(a[n-1:, n-1])
        utx = 0
        for p in range(len(u)):
            utx += u[p]*a[n-1+p, n-1]
        a[n-1:, n-1] -= beta*utx*u
        for i in range (m):
            utx = 0
            for p in range(len(u)):
                utx += u[p]*Qt[n-1+p, i]
            Qt[n-1:, i] -= beta*utx*u
    q = np.transpose(Qt)
    r = a
    return q, r

# 
# Mètodes iteratius
#

# Càlcul del radi espectral de la matriu A 

# Funció: radi_espectral
def radi_espectral(A):

    v = np.linalg.eigvals(A)
    rho = np.abs(v[0])
    for i in range(1, len(A)):
        if np.abs(v[i]) > rho:
            rho = np.abs(v[i])
    return rho

# Funció: jacobi
def jacobi(A, b, x, eps, nitm):

    n = len(A)
    #producte inv_D@A
    J = np.copy(A)
    for i in range(n):
        J[i] /= A[i, i]
    J = np.eye(n)-J
    nit = 0
    r = b - A@x
    #per aumentar eficiencia, els valors constants com inv_D@B o eps*norm_b el calculare una vegada nomes en una constant
    inv_Db = np.zeros((n, 1))
    for i in range(n):
        inv_Db[i, 0] = b[i, 0]/A[i, i]
    error = np.linalg.norm(b)*eps
    while np.linalg.norm(r) > error and nit < nitm:
        nit += 1
        x = J@x+inv_Db
        r = b - A@x
    if nit >= nitm:
        nit = -1
    return x, norm(r, ord = 2), nit

# Funció: gauss_seidel
def gauss_seidel(A, b, x, eps, nitm):
    
    n = len(A)
    nit = 0
    r = b-A@x
    error = np.linalg.norm(b)*eps
    while np.linalg.norm(r) > error and nit < nitm:
        nit += 1
        for i in range(n):
            s = 0
            for p in range(i):
                s += A[i, p]*x[p, 0]
            for p in range(i+1 ,n):
                s += A[i, p]*x[p, 0]
            x[i, 0] = b[i, 0]-s
            x[i, 0] /= A[i, i]
        r = b - A@x
    if nit >= nitm:
        nit = -1
    return x, norm(r, ord = 2), nit 

#Funció: sor
def sor(A, b, x, w, eps, nitm):
    
    n = len(A)
    nit = 0
    r = b-A@x
    error = np.linalg.norm(b)*eps
    while np.linalg.norm(r) > error and nit < nitm:
        nit += 1
        for i in range(n):
            s = 0
            for p in range(n):
                s += A[i, p]*x[p, 0]
            s = w*(b[i, 0] - s)/A[i, i]
            x[i, 0] += s
        r = b - A@x
    if nit >= nitm:
        nit = -1
    return x, norm(r, ord = 2), nit 

# Funció: gradient
def gradient(A, b, x, ε, nitm):
    
    n = len(A)
    nit = 0
    r = b - A@x
    error = np.linalg.norm(b)*ε
    iterats = []
    while (np.linalg.norm(r) > error and nit < nitm):
        nit += 1
        Ar = A@r
        alfa = np.dot(r.flatten(), r.flatten())/np.dot(r.flatten(), (Ar).flatten())
        x += alfa*r
        iterats.append(x.copy())
        r -= alfa * Ar
    if nit >= nitm:
        nit = -1
    norma_residu = np.linalg.norm(r)
    iterats = np.array(iterats)
    iterats = iterats.reshape(nit, n)
    iterats = np.transpose(iterats)
    return x, norma_residu, nit, iterats 

# Funció: gradient_conjugat
def gradient_conjugat(A, b, x, ε, nitm):
    
    n = len(A)
    nit = 0
    r = b - A@x
    p = np.zeros((n, 1))
    beta = 0
    error = np.linalg.norm(b)*ε
    iterats = []
    while (np.linalg.norm(r) > error and nit < nitm):
        nit += 1
        p = r + beta*p
        Ap = A@p #nomes ho calculem una vegada
        ptAp = np.dot(p.flatten(), (Ap).flatten()) #nomes ho calculem una vegada
        alpha = np.dot(p.flatten(), r.flatten())/ptAp
        x += alpha * p
        r = r - alpha*(Ap) #aixi disminuira l'error
        beta = -np.dot(r.flatten(), (Ap).flatten())/ptAp
        iterats.append(x.copy())
    norma_residu = np.linalg.norm(r)
    if nit >= nitm:
        nit = -1
    iterats = np.array(iterats)
    iterats = iterats.reshape(nit, n)
    iterats = np.transpose(iterats)
    return x, norma_residu, nit, iterats

#
# Càlcul de valors propis (VAPs) i vectors propis (VEPs) 
#

# Funció: metode_de_la_potencia
def metode_de_la_potencia(a, z, tol, maxit):
    
    n = len(a)
    y = z/norm(z)
    z = a@y
    vap = float('inf') #ho fem per a evitar que per casualitat, el vap_nou s'apropi molt a la inicialitzacio de vap en la primera iteracio
    i0 =  argmax(abs(y)) #Per evitar errors desmesurats, volem que el denominador de la divisio sigui el mes gran possible, dins el vector normalitzat
    vap_nou = z[i0]/y[i0]
    nit = 1
    while abs(vap_nou - vap) >= tol*abs(vap_nou) and nit < maxit:
        nit += 1
        vap = vap_nou #li donem el valor de la iteracio anterior
        y = z/norm(z)
        z = a@y
        i0 =  argmax(abs(y)) #Per evitar errors desmesurats, volem que el denominador de la divisio sigui el mes gran possible, dins el vector normalitzat
        vap_nou = z[i0]/y[i0]
    if nit >= maxit:
        nit = -1
    vap = vap_nou
    vep = y
    return vap, vep, nit 

# Funció: metode_de_la_potencia_inversa
def metode_de_la_potencia_inversa(a, z, tol, maxit):
    
    n = len(a)
    y = z/norm(z)
    LU, p = lu_ppe(a.copy()) #hem de resoldre el sistema Az = y, per trobar z(k+1), aqui modifiquem la matriu a, s'ha de tenir en compte al usar la funcio
    z = sol_lu(LU, y[p], True)
    vap = float('inf') #ho fem per a evitar que per casualitat, el vap_nou s'apropi molt a la inicialitzacio de vap en la primera iteracio
    i0 =  argmax(abs(y)) #Per evitar errors desmesurats, volem que el denominador de la divisio sigui el mes gran possible, dins el vector normalitzat
    vap_nou = z[i0]/y[i0]
    nit = 1
    while abs(vap_nou - vap) >= tol*abs(vap_nou) and nit < maxit:
        nit += 1
        vap = vap_nou #li donem el valor de la iteracio anterior
        y = z/norm(z)
        z = sol_lu(LU, y[p], True)
        i0 =  argmax(abs(y)) #Per evitar errors desmesurats, volem que el denominador de la divisio sigui el mes gran possible, dins el vector normalitzat
        vap_nou = z[i0]/y[i0]
    if nit >= maxit:
        nit = -1
    vap = 1/vap_nou
    vep = y
    return vap, vep, nit 

# Funcióa: jacobi_classic()
def jacobi_classic(a,tol,maxit):
    
    a = a.copy()
    n = len(a)
    c = eye(n) #Per obtenir la matriu ortogonal basicament aplicarem la rotacio a la identitat per la dreta, pues Q(k) = Q(k-1)*R, i Q(0) = Id
    M = abs(a)
    M[diag_indices(n)] = 0  #ens construim la matriu a sense la diagonal
    p, q = unravel_index(argmax(M), (n, n)) #ens diu la posicio del element maxim en abs. Sempre hi ha 2 per simetria, per tant es quedara amb el de la part triangular superior (j > i). 
    nit = 0
    while M[p, q] >= tol and nit < maxit:
        nit += 1
        eta = (a[p, p] - a[q, q])/(2*a[p, q])
        if eta < 0:
            tg = -eta + (eta**2 + 1)**0.5
        else:
            tg = -eta - (eta**2 + 1)**0.5
        cos = 1/(1+tg**2)**0.5
        sin = cos*tg
        col_p = c[:, p].copy() #guardem les columnes p i q per a que no es sobreescriguin el fer les operacions de rotacio
        col_q = c[:, q].copy()
        c[:, p] = cos * col_p + sin * col_q
        c[:, q] = -sin * col_p + cos * col_q
        fila_p = a[p, :].copy()
        fila_q = a[q, :].copy()
        a[p, p] = a[p, p]*cos**2 + 2*a[p, q] * sin * cos + a[q, q] * sin**2 #encara no s'ha sobreescrit res, per tant uso l'original
        a[q, q] = fila_p[p]*sin**2 - 2*a[p, q]*sin*cos + a[q, q]*cos**2 #encara no s'ha sobreescrit a[p, q]
        a[p, q] = 0 #trobem l'angle tq s'anulin aquests termes
        a[q, p] = 0
        for r in range (n):
            if r != p and r != q:
                a[p, r] = fila_p[r]*cos + fila_q[r]*sin
                a[r, p] = a[p, r]
                a[q, r] = -fila_p[r]*sin + fila_q[r] * cos
                a[r, q] = a[q, r]
        #Ara que he acabat la iteracio, tornem a calcular l'element mes gran de la matriu sense la diagonal per a tornar a entrar el while
        M = abs(a)
        M[diag_indices(n)] = 0  #ens construim la matriu a sense la diagonal
        p, q = unravel_index(argmax(M), (n, n))
    if nit >= maxit:
        nit = -1
    d = a
    return d, c, nit 

# Funció givens
def givens(a):
    
    a = a.copy() 
    n = len(a)
    rot = eye(n) #basicament aplicarem les rotacions a la identitat (només per la dreta) per a obtenir rot.
    for p in range (1, n-1):
        for q in range (p+1, n):
            if a[q, p-1] != 0:
                #Hem de trobar l'angle de la rotacio
                cte = (a[q, p-1]**2 + a[p, p-1]**2)**0.5
                sin = a[q, p-1]/cte
                cos = a[p, p-1]/cte
                #Ara hem d'aplicar la rotacio a la matriu i a la matriu c (nomes per la dreta)
                col_p = rot[:, p].copy() #guardem les columnes p i q per a que no es sobreescriguin el fer les operacions de rotacio
                col_q = rot[:, q].copy()
                rot[:, p] = cos * col_p + sin * col_q
                rot[:, q] = -sin * col_p + cos * col_q
                fila_p = a[p, :].copy()
                fila_q = a[q, :].copy()
                a[p, p] = a[p, p]*cos**2 + 2*a[p, q] * sin * cos + a[q, q] * sin**2 #encara no s'ha sobreescrit res, per tant uso l'original
                a[q, q] = fila_p[p]*sin**2 - 2*a[p, q]*sin*cos + a[q, q]*cos**2 #encara no s'ha sobreescrit a[p, q]
                a[p, q] = a[p, q]*cos**2 + (fila_q[q] - fila_p[p])*cos*sin - a[p, q]*sin**2
                a[q, p] = a[p, q]
                for r in range (n):
                    if r != p and r != q:
                        a[p, r] = fila_p[r]*cos + fila_q[r]*sin
                        a[r, p] = a[p, r]
                        a[q, r] = -fila_p[r]*sin + fila_q[r] * cos
                        a[r, q] = a[q, r]
    h = a
    return h, rot

# Funció qr_hessenberg
def qr_hessenberg(a,tol,itmax):
    
    a = a.copy() 
    n = len(a)
    Q = eye(n)
    num_it = 0
    while max(abs(diag(a, k = -1))) >= tol and num_it < itmax:
        num_it += 1
        # Creem llistes per guardar els sinus i cosinus d'aquesta iteracio
        cos_list = []
        sin_list = []
        for j in range (0, n-1):
            # Hem de calcular el sin i cos de cada rotacio
            cte = (a[j, j]**2 + a[j+1, j]**2)**0.5
            sin = -a[j+1, j]/cte
            cos = a[j, j]/cte
            cos_list.append(cos)
            sin_list.append(sin)
            fila_p = a[j, :].copy() #guardem les files p i q per a que no es sobreescriguin el fer les operacions de rotacio
            fila_q = a[j+1, :].copy()
            a[j, :] = cos*fila_p - sin*fila_q # Multipliquem la matriu de rotacio per l'esquerra, per tant nomes canvien les files p i q
            a[j+1, :] = sin*fila_p + cos*fila_q
            # A la matriu Q multipliquem les matrius de rotacio per la dreta, pero amb l'angle negatiu (sin --> -sin, cos --> cos)
            col_p = Q[:, j].copy() #guardem les columnes p i q per a que no es sobreescriguin al fer les operacions de rotacio
            col_q = Q[:, j+1].copy()
            Q[:, j] = cos * col_p - sin * col_q
            Q[:, j+1] = sin * col_p + cos * col_q
        for j in range (0, n-1): # Multipliquem A per Q
            col_p = a[:, j].copy()
            col_q = a[:, j+1].copy()
            a[:, j] = cos_list[j] * col_p - sin_list[j] * col_q
            a[:, j+1] = sin_list[j] * col_p + cos_list[j] * col_q
        # Ara tenim que la matriu a = R*Q, per tant, hem de tornar a iterar
    if num_it >= itmax:
        num_it = -1
    T = a
    return T, Q, num_it