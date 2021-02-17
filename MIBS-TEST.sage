

import time;

#multimesage
Nr=5;
Nm=16;

print( 'Comparison of Equation speed solving.\n Nr=',Nr,', Nm=',Nm);
print('For PolyBoRi MIBS-FWBW.')

    
names = [];


names += map_threaded(lambda u: 'K_'+str(u),range(80+Nr*8));
for i in range(Nm):
    for j in range(Nr+1):
        names += map_threaded(lambda u: 'L_'+str(i)+'_'+str(j)+'_'+str(u),range(32));

#for i in range(Nm):
#        names += map_threaded(lambda u: 'X_'+str(i)+'_'+str(u),range(64));

#for i in range(Nm):
#        names += map_threaded(lambda u: 'Y_'+str(i)+'_'+str(u),range(64));

namesK = map_threaded(lambda u: 'K_'+str(u),range(80)); 

#orig box
def MibsBox(X):
    return vector(ring,[ 
    X[0]*X[1]*X[2] + X[0]*X[1]*X[3] + X[0]*X[3] + X[0] + X[1]*X[2]*X[3] + X[1]*X[3] + X[1] + X[2] + X[3],
    X[0]*X[1]*X[2] + X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0] + X[1]*X[2] + X[1]*X[3] + X[1] + X[3],
    X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0]*X[2] + X[0]*X[3] + X[1]*X[2] + X[1] + X[3] + 1, 
    X[0]*X[1]*X[2] + X[0]*X[2] + X[0] + X[1]*X[2]*X[3] + X[1]*X[3] + X[2] + X[3] 
    ]);

    
def MibsBoxAdd(X,Y):
    return [
        X[0] + Y[0]*Y[1]*Y[2] + Y[0]*Y[1] + Y[0]*Y[3] + Y[0] + Y[1]*Y[2]*Y[3] + Y[1]*Y[3] + Y[1] + Y[2] + 1, 
X[1] + Y[0]*Y[1]*Y[3] + Y[0]*Y[1] + Y[0]*Y[2]*Y[3] + Y[1]*Y[2] + Y[1] + Y[2]*Y[3] + Y[2] + 1, 
X[2] + Y[0]*Y[1]*Y[3] + Y[0]*Y[1] + Y[0]*Y[2]*Y[3] + Y[0]*Y[3] + Y[1]*Y[2]*Y[3] + Y[1]*Y[2] + Y[1]*Y[3] + Y[2] + Y[3] + 1, 
X[3] + Y[0]*Y[1]*Y[2] + Y[0]*Y[2]*Y[3] + Y[0] + Y[1]*Y[2]*Y[3] + Y[1]*Y[3] + Y[1],
Y[0] + X[0]*X[1]*X[2] + X[0]*X[1]*X[3] + X[0]*X[3] + X[0] + X[1]*X[2]*X[3] + X[1]*X[3] + X[1] + X[2] + X[3],
Y[1] + X[0]*X[1]*X[2] + X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0] + X[1]*X[2] + X[1]*X[3] + X[1] + X[3],
Y[2] + X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0]*X[2] + X[0]*X[3] + X[1]*X[2] + X[1] + X[3] + 1, 
Y[3] + X[0]*X[1]*X[2] + X[0]*X[2] + X[0] + X[1]*X[2]*X[3] + X[1]*X[3] + X[2] + X[3]
];



def MibsLin(X):
    T = copy(X);
    T[7*4:8*4] = X[6*4:7*4]+X[5*4:6*4]+X[4*4:5*4]+X[3*4:4*4]+X[2*4:3*4]+X[1*4:2*4];
    T[6*4:7*4] = X[7*4:8*4]+X[5*4:6*4]+X[4*4:5*4]+X[2*4:3*4]+X[1*4:2*4]+X[0*4:1*4];
    T[5*4:6*4] = X[7*4:8*4]+X[6*4:7*4]+X[4*4:5*4]+X[3*4:4*4]+X[1*4:2*4]+X[0*4:1*4];
    T[4*4:5*4] = X[7*4:8*4]+X[6*4:7*4]+X[5*4:6*4]+X[3*4:4*4]+X[2*4:3*4]+X[0*4:1*4];
    T[3*4:4*4] = X[7*4:8*4]+X[6*4:7*4]+X[4*4:5*4]+X[3*4:4*4]+X[2*4:3*4];
    T[2*4:3*4] = X[7*4:8*4]+X[6*4:7*4]+X[5*4:6*4]+X[2*4:3*4]+X[1*4:2*4];
    T[1*4:2*4] = X[6*4:7*4]+X[5*4:6*4]+X[4*4:5*4]+X[1*4:2*4]+X[0*4:1*4];
    T[0*4:1*4] = X[7*4:8*4]+X[5*4:6*4]+X[4*4:5*4]+X[3*4:4*4]+X[0*4:1*4];
    return T;    

def MibsInvLin(X):
    T = copy(X);
    T[7*4:8*4] = X[7*4:8*4]+X[6*4:7*4]+X[3*4:4*4]+X[2*4:3*4]+X[1*4:2*4];
    T[6*4:7*4] = X[6*4:7*4]+X[5*4:6*4]+X[2*4:3*4]+X[1*4:2*4]+X[0*4:1*4];
    T[5*4:6*4] = X[5*4:6*4]+X[4*4:5*4]+X[3*4:4*4]+X[1*4:2*4]+X[0*4:1*4];
    T[4*4:5*4] = X[7*4:8*4]+X[4*4:5*4]+X[3*4:4*4]+X[2*4:3*4]+X[0*4:1*4];
    T[3*4:4*4] = X[7*4:8*4]+X[6*4:7*4]+X[4*4:5*4]+X[2*4:3*4]+X[1*4:2*4]+X[0*4:1*4];
    T[2*4:3*4] = X[7*4:8*4]+X[6*4:7*4]+X[5*4:6*4]+X[3*4:4*4]+X[1*4:2*4]+X[0*4:1*4];
    T[1*4:2*4] = X[6*4:7*4]+X[5*4:6*4]+X[4*4:5*4]+X[3*4:4*4]+X[2*4:3*4]+X[0*4:1*4];
    T[0*4:1*4] = X[7*4:8*4]+X[5*4:6*4]+X[4*4:5*4]+X[3*4:4*4]+X[2*4:3*4]+X[1*4:2*4];
    return T;           

def MibsPerm(X):
    T = copy(X);
    T[6*4:7*4] = X[7*4:8*4];
    T[0*4:1*4] = X[6*4:7*4];
    T[7*4:8*4] = X[5*4:6*4];
    T[5*4:6*4] = X[4*4:5*4];
    T[2*4:3*4] = X[3*4:4*4];
    T[1*4:2*4] = X[2*4:3*4];
    T[4*4:5*4] = X[1*4:2*4];
    T[3*4:4*4] = X[0*4:1*4];
    return T;

    
def MibsInvPerm(X):
    T = copy(X);
    T[7*4:8*4] = X[6*4:7*4];
    T[6*4:7*4] = X[0*4:1*4];
    T[5*4:6*4] = X[7*4:8*4];
    T[4*4:5*4] = X[5*4:6*4];
    T[3*4:4*4] = X[2*4:3*4];
    T[2*4:3*4] = X[1*4:2*4];
    T[1*4:2*4] = X[4*4:5*4];
    T[0*4:1*4] = X[3*4:4*4];
    return T;

def MibsF(L,K):
    T=copy((L+K));
    T[0*4:1*4] = MibsBox(T[0*4:1*4]);
    T[1*4:2*4] = MibsBox(T[1*4:2*4]);
    T[2*4:3*4] = MibsBox(T[2*4:3*4]);
    T[3*4:4*4] = MibsBox(T[3*4:4*4]);
    T[4*4:5*4] = MibsBox(T[4*4:5*4]);
    T[5*4:6*4] = MibsBox(T[5*4:6*4]);
    T[6*4:7*4] = MibsBox(T[6*4:7*4]);
    T[7*4:8*4] = MibsBox(T[7*4:8*4]);
#    print 'after box: ', hex(ZZ(list(T),base=2));
    T = MibsLin(T);
    T = MibsPerm(T);
#    print 'after lin: ', T;
    return T;

def MibsKeyRoundEqs(rnd, Stt, AllKeys):
    T = copy(Stt);
    t = rnd+64+1;
    tt = t.bits();
    rc = vector(GF(2),tt[0:5]);
    for i in range(80):
        T[i] = Stt[(i+19)%80];
    keqs = list(AllKeys[80+rnd*8+4:80+rnd*8+8] + MibsBox(T[76:80]));
    keqs += list(AllKeys[80+rnd*8:80+rnd*8+4] + MibsBox(T[72:76]));    
    T[76:80] = AllKeys[80+rnd*8+4:80+rnd*8+8]
    T[72:76] = AllKeys[80+rnd*8:80+rnd*8+4];
    T[14:19] = T[14:19]+rc;
    return [T[48:80],T,keqs];    

def MibsKeyEqs(ring,KS,rnd):
    eqs = [];
    AllKeys = vector(map_threaded(lambda v:ring('K'+pad_zeros(v)),range(80+8*Nr)));
    Stt = copy(KS);
    rk = Stt[48:80];
    RK=[];
    for i in range(rnd):
        [rk,Stt,keqs]=MibsKeyRoundEqs(i,Stt,AllKeys);
        RK+=[rk];
        eqs += keqs;
    return RK,eqs;


def MibsEncryptEqs(ring,numKP,RK,X,rnd,Y):
    eqs = [];
    L = map_threaded(lambda u:vector(map_threaded(lambda v:ring('L_'+str(numKP)+'_'+pad_zeros(u*32+v)),range(32))),range(rnd+1));
    L0=copy(X[32:64]);
    R0=copy(X[0:32]);
    OL=copy(Y[32:64]);
    OR=copy(Y[0:32])
    L += [R0];
    L[0] = copy(L0);
    #L[-1] = copy(R0);
    L[rnd-1] = copy(OL);
    L[rnd] = copy(OR);
    for i in range(rnd):
        eqs += list(MibsF(L[i],RK[i])+L[i-1]+L[i+1]);
    return eqs;
        
def MibsKeyRoundEqsMQ(rnd, Stt, AllKeys):
    T = copy(Stt);
    t = rnd+64+1;
    tt = t.bits();
    rc = vector(GF(2),tt[0:5]);
    for i in range(80):
        T[i] = Stt[(i+19)%80];
    keqs = MibsBoxAdd(T[76:80],AllKeys[80+rnd*8+4:80+rnd*8+8]);
    keqs += MibsBoxAdd(T[72:76],AllKeys[80+rnd*8:80+rnd*8+4]);    
    T[76:80] = AllKeys[80+rnd*8+4:80+rnd*8+8]
    T[72:76] = AllKeys[80+rnd*8:80+rnd*8+4];
    T[14:19] = T[14:19]+rc;
    return [T[48:80],T,keqs]; 

def MibsKeyEqsMQ(ring,KS,rnd):
    eqs = [];
    Stt = copy(KS);
    AllKeys = vector(map_threaded(lambda v:ring('K'+pad_zeros(v)),range(80+8*Nr)));
    Stt = copy(KS);
    rk = Stt[48:80];
    RK=[];
    for i in range(rnd):
        [rk,Stt,keqs]=MibsKeyRoundEqsMQ(i,Stt,AllKeys);
        RK+=[rk];
        eqs += keqs;
    return RK,eqs;

def MibsEncryptEqsMQ(ring,numKP,RK,X,rnd,Y):
    eqs = [];
    L = map_threaded(lambda u:vector(map_threaded(lambda v:ring('L_'+str(numKP)+'_'+pad_zeros(u*32+v)),range(32))),range(rnd+1));
    L0=copy(X[32:64]);
    R0=copy(X[0:32]);
    OL=copy(Y[32:64]);
    OR=copy(Y[0:32])
    
    L += [R0];
    L[0] = copy(L0);
    L[rnd-1] = copy(OL);
    L[rnd] = copy(OR);
    for i in range(rnd):
        TT=copy(L[i]+RK[i]);
        T = L[i-1] + L[i+1];
        T = MibsInvPerm(T);
        T = MibsInvLin(T);
        teqs=[];
        teqs+=MibsBoxAdd(TT[0*4:1*4],T[0*4:1*4]);
        teqs+=MibsBoxAdd(TT[1*4:2*4],T[1*4:2*4]);
        teqs+=MibsBoxAdd(TT[2*4:3*4],T[2*4:3*4]);
        teqs+=MibsBoxAdd(TT[3*4:4*4],T[3*4:4*4]);
        teqs+=MibsBoxAdd(TT[4*4:5*4],T[4*4:5*4]);
        teqs+=MibsBoxAdd(TT[5*4:6*4],T[5*4:6*4]);
        teqs+=MibsBoxAdd(TT[6*4:7*4],T[6*4:7*4]);
        teqs+=MibsBoxAdd(TT[7*4:8*4],T[7*4:8*4]);
        eqs += list(ideal(teqs).interreduced_basis());
    return eqs;

def AIOMibsEncryptEqsMQ(ring,KP,rnd,KS,X,Y,idx):
    eqs = rnd*[0];
    Stt = copy(KS);
    AllKeys = vector(map_threaded(lambda v:ring('K'+pad_zeros(v)),range(80+8*Nr)));
    L=KP*[0];
    for i in range(KP):
        L[i] = map_threaded(lambda u:vector(map_threaded(lambda v:ring('L_'+str(idx+i)+'_'+pad_zeros(u*32+v)),range(32))),range(rnd+1));
        
    L0 = KP*[0];
    R0 = KP*[0];
    OL = KP*[0];
    OR = KP*[0];
    for i in range(KP):
        L0[i]=copy(X[i][32:64]);
        R0[i]=copy(X[i][0:32]);
        OL[i]=copy(Y[i][32:64]);
        OR[i]=copy(Y[i][0:32])
    for i in range(KP):
        L[i] += [R0[i]];
        L[i][0] = copy(L0[i]);
        L[i][rnd-1] = copy(OL[i]);
        L[i][rnd] = copy(OR[i]);
    
    for i in range(rnd):
        eqs[i]=[];
        reqs= [];
        [rk,Stt,keqs]=MibsKeyRoundEqsMQ(i,Stt,AllKeys);
        reqs+=keqs;
        for j in range(KP):
            TT=copy(L[j][i]+rk);
            T = L[j][i-1] + L[j][i+1];
            T = MibsInvPerm(T);
            T = MibsInvLin(T);
            teqs=[];
            teqs+=MibsBoxAdd(TT[0*4:1*4],T[0*4:1*4]);
            teqs+=MibsBoxAdd(TT[1*4:2*4],T[1*4:2*4]);
            teqs+=MibsBoxAdd(TT[2*4:3*4],T[2*4:3*4]);
            teqs+=MibsBoxAdd(TT[3*4:4*4],T[3*4:4*4]);
            teqs+=MibsBoxAdd(TT[4*4:5*4],T[4*4:5*4]);
            teqs+=MibsBoxAdd(TT[5*4:6*4],T[5*4:6*4]);
            teqs+=MibsBoxAdd(TT[6*4:7*4],T[6*4:7*4]);
            teqs+=MibsBoxAdd(TT[7*4:8*4],T[7*4:8*4]);
            reqs += teqs;
                    
        eqs[i] = reqs;
        #if(i<floor((Nr+1)/2)):
        #    eqs = list(ideal(eqs).interreduced_basis());
    up_eqs=[];
    dn_eqs=[];
    j=rnd-1;
    for i in range(rnd):
        if(i<floor((Nr+1)/2)):
            up_eqs += eqs[i];
            up_eqs = list(ideal(up_eqs).interreduced_basis());
        else:
            dn_eqs += eqs[j];
            #dn_eqs = list(ideal(dn_eqs).interreduced_basis());
            j=j-1;
        print (i,'-------')
    return up_eqs+dn_eqs;        
    #return eqs;    
    
def MibsKeyRound(Stt, rc):
    T = copy(Stt);
    for i in range(80):
        T[i] = Stt[(i+19)%80];
    T[76:80] = MibsBox(T[76:80]);
    T[72:76] = MibsBox(T[72:76]);
    T[14:19] = T[14:19]+rc;
    return [T[48:80],T]; 
        
def MibsEncrypt(KS,X,rnd):
    print(X)
    Stt = copy(KS);
    L=copy(X[32:64]);
    R=copy(X[0:32]);
    for i in range(rnd):
        t = i+64+1;
        tt = t.bits();
        rc = vector(GF(2),tt[0:5]);
        [rk,Stt]=MibsKeyRound(Stt, rc);
        T = MibsF(L,rk);
        print('rnd:',i,'  l: ',L);
        print('rnd:',i,'  k: ',rk);
        T = T + R;
        R = copy(L);
        L = copy(T);
    
    Y=copy(X);
    Y[0:32]=L;
    Y[32:64]=R;
    return Y;

def MibsDecrypt(KS,X,rnd):
    Stt = copy(KS);
    rk = rnd*[vector(GF(2),32*[0])];

    L=copy(X[32:64]);
    R=copy(X[0:32]);
    
    for i in range(rnd):
        t = i+64+1;
        tt = t.bits();
        rc = vector(GF(2),tt[0:5]);
        [rk[i],Stt]=MibsKeyRound(Stt, rc);
    
    for i in range(rnd):
        T = MibsF(L,rk[rnd-i-1]);
        T = T + R;
        R = copy(L);
        L = copy(T);
    
    Y=copy(X);
    Y[0:32]=L;
    Y[32:64]=R;
    return Y;
    
from sage.rings.polynomial.pbori import GroebnerStrategy, ReductionStrategy

patt=[0,1,2,3,4,5,6,7];
def num_to_patt(n,patt):
    msg = vector(GF(2),64);
    nn=Integer(n).digits(base=2,padto=64);
    for i in range(64):
        if nn[i]:
            msg[patt[i]]=1;
    return msg;

    
print ('Check Implementation')
ring =BooleanPolynomialRing(names=names,order='degrevlex');
KS = vector(ring,Integer(0x0).digits(base=2,padto=80));
X = Nm*[0];
Y = Nm*[0];
for i in range(Nm):
    X[i] = num_to_patt(i,patt);
    Y[i]=MibsEncrypt(KS,X[i],Nr)
#print X;


print (ring.ngens());

print( 'Start experiments...');

#XX = [];
#YY = [];
#for i in range(Nm):
#    XX += [ring('X_'+str(i)+'_'+str(j))+X[i][j] for j in range(64)];
#    YY += [ring('Y_'+str(i)+'_'+str(j))+Y[i][j] for j in range(64)];

#KK = [ring('K_'+str(i))+KS[i] for i in range(80)];

print("Reading Equations");
eqs=[];
file = open('eqs.txt', 'r')
Lines = file.readlines()
for line in Lines:
    try:
    	p=ring(line);
    except:
        continue;
    #print (p);
    eqs += [p];

print("Solving Equations");
print (list(ideal(eqs).groebner_basis(prot=True,faugere=False,linear_algebra_in_last_block=False,selection_size=10000))) 
    

