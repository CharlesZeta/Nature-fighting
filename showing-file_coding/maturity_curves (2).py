#!/usr/bin/env python3
"""
Option Price vs Time-to-Maturity: DST (Rough Heston + Dual Hawkes), Merton, BSM
  - European & American Put/Call
  - Multiple alpha (gamma) for DST
  - Control variate diagnostics (correlated GBM paths)
"""

import math, time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import warnings; warnings.filterwarnings('ignore')

np.random.seed(2024)

# ============================================================
# 1. BSM
# ============================================================
def bsm(S,K,T,r,sig,opt='put'):
    if T < 1e-10:
        return max(K-S,0) if opt=='put' else max(S-K,0)
    d1 = (np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = d1-sig*np.sqrt(T)
    if opt=='call': return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

# ============================================================
# 2. Merton European (series)
# ============================================================
def merton_euro(S,K,T,r,sig,lam,mJ,sJ,opt='put',Nt=30):
    if T < 1e-10:
        return max(K-S,0) if opt=='put' else max(S-K,0)
    ms = np.exp(mJ+.5*sJ**2)-1; p=0.
    for n in range(Nt):
        sn = np.sqrt(sig**2+n*sJ**2/T)
        rn = r-lam*ms+n*np.log(1+ms)/T
        lp = lam*(1+ms)
        w = np.exp(-lp*T)*(lp*T)**n/math.factorial(n)
        p += w*bsm(S,K,T,rn,sn,opt)
    return p

# ============================================================
# 3. Merton MC paths
# ============================================================
def merton_paths(S0,r,sig,lam,mJ,sJ,T,Ns,Np):
    dt=T/Ns; ms=np.exp(mJ+.5*sJ**2)-1
    S=np.zeros((Np,Ns+1)); S[:,0]=S0
    for i in range(Ns):
        Z=np.random.randn(Np)
        Nj=np.random.poisson(lam*dt,Np)
        Jsum=Nj*mJ+np.sqrt(np.maximum(Nj,0))*sJ*np.random.randn(Np)
        Jsum[Nj==0]=0.
        S[:,i+1]=S[:,i]*np.exp((r-lam*ms-.5*sig**2)*dt+sig*np.sqrt(dt)*Z+Jsum)
    return S

# ============================================================
# 4. DST MC paths (Rough Heston + Dual Hawkes)
# ============================================================
def gl_nodes(Nexp,alpha):
    from numpy.polynomial.laguerre import laggauss
    from scipy.special import gamma as gf
    nd,wt=laggauss(Nexp)
    om=wt*nd**(alpha-1)/gf(alpha)
    om=np.maximum(om,1e-14); om/=om.sum()
    return np.maximum(nd,1e-6),om

def dst_paths(S0,r,V0,kap,th,xi,rho,alpha,
              liS,liV,bS,bV,eSS,eVV,eSV,eVS,
              mJ,sJ,mV,T,Ns,Np,Nexp=6,
              return_gbm=False,sig_gbm=0.2):
    dt=T/Ns; sdt=np.sqrt(dt)
    xl,wl=gl_nodes(Nexp,alpha)
    ms=np.exp(mJ+.5*sJ**2)-1
    S=np.zeros((Np,Ns+1)); S[:,0]=S0
    Va=np.zeros((Np,Ns+1)); Va[:,0]=V0
    U=np.zeros((Np,Nexp))
    lS=np.full(Np,liS); lV=np.full(Np,liV)
    if return_gbm:
        Sg=np.zeros((Np,Ns+1)); Sg[:,0]=S0
    for i in range(Ns):
        Vp=np.maximum(Va[:,i],0.); sqV=np.sqrt(Vp)
        Z1=np.random.randn(Np); Z2=np.random.randn(Np)
        dWs=Z1*sdt; dWv=rho*dWs+np.sqrt(1-rho**2)*Z2*sdt
        pS=np.minimum(lS*dt,.99)
        mSk=(np.random.random(Np)<pS).astype(float)
        JS=np.where(mSk>0,np.random.normal(mJ,sJ,Np),0.)
        pV=np.minimum(lV*dt,.99)
        mVj=(np.random.random(Np)<pV).astype(float)
        JV=np.where(mVj>0,np.random.exponential(mV,Np),0.)
        for l in range(Nexp):
            U[:,l]+=(-xl[l]*U[:,l]+kap*(th-Vp))*dt+xi*sqV*dWv+JV*mVj
        Va[:,i+1]=V0+(wl[None,:]*U).sum(1)
        lS+=bS*(liS-lS)*dt+eSS*mSk+eSV*mVj
        lV+=bV*(liV-lV)*dt+eVV*mVj+eVS*mSk
        lS=np.maximum(lS,1e-8); lV=np.maximum(lV,1e-8)
        Vn=np.maximum(Va[:,i+1],0.); Vm=.5*(Vp+Vn)
        sqVm=np.sqrt(np.maximum(Vm,0.))
        S[:,i+1]=S[:,i]*np.exp((r-lS*ms-.5*Vm)*dt+sqVm*dWs)*(1+(np.exp(JS)-1)*mSk)
        if return_gbm:
            Sg[:,i+1]=Sg[:,i]*np.exp((r-.5*sig_gbm**2)*dt+sig_gbm*dWs)
    if return_gbm: return S,Va,Sg
    return S,Va

# ============================================================
# 5. GBM paths
# ============================================================
def gbm_paths(S0,r,sig,T,Ns,Np):
    dt=T/Ns; S=np.zeros((Np,Ns+1)); S[:,0]=S0
    for i in range(Ns):
        S[:,i+1]=S[:,i]*np.exp((r-.5*sig**2)*dt+sig*np.sqrt(dt)*np.random.randn(Np))
    return S

# ============================================================
# 6. LSM & MC
# ============================================================
def lsm(S,K,r,T,opt='put'):
    Np,Ns1=S.shape; Ns=Ns1-1; dt=T/Ns
    pf=lambda s: np.maximum(K-s,0.) if opt=='put' else np.maximum(s-K,0.)
    cf=pf(S[:,-1]); ex_t=np.full(Np,Ns)
    for t in range(Ns-1,0,-1):
        cf*=np.exp(-r*dt)
        iv=pf(S[:,t]); itm=iv>0
        if itm.sum()<10: continue
        X=S[itm,t]/K
        A=np.column_stack([np.ones(itm.sum()),X,X**2,X**3])
        try:
            c=np.linalg.lstsq(A,cf[itm],rcond=None)[0]
            cont=A@c
        except: continue
        ex=iv[itm]>cont; idx=np.where(itm)[0][ex]
        cf[idx]=iv[idx]; ex_t[idx]=t
    pv=np.array([pf(S[i,ex_t[i]])*np.exp(-r*ex_t[i]*dt) for i in range(Np)])
    return pv.mean()

def mc_euro(S,K,r,T,opt='put'):
    if opt=='put': pay=np.maximum(K-S[:,-1],0.)
    else:          pay=np.maximum(S[:,-1]-K,0.)
    return np.exp(-r*T)*pay.mean()

def mc_euro_payoffs(S,K,r,T,opt='put'):
    if opt=='put': pay=np.maximum(K-S[:,-1],0.)
    else:          pay=np.maximum(S[:,-1]-K,0.)
    return np.exp(-r*T)*pay

def cv_estimate(pay_m, pay_g, bsm_ex):
    cov_mat=np.cov(pay_m,pay_g)
    if cov_mat[1,1]<1e-15:
        return pay_m.mean(), pay_m.std()/np.sqrt(len(pay_m))
    beta=cov_mat[0,1]/cov_mat[1,1]
    adj=pay_m-beta*(pay_g-bsm_ex)
    return adj.mean(), adj.std()/np.sqrt(len(adj))

# ============================================================
# 7. Main
# ============================================================
def main():
    t0=time.time()
    S0=100.; r=0.03; K_put=100.; K_call=100.
    Ns_per_year=200  # steps per year of maturity
    Np=18000

    # Maturity grid
    Ts = np.array([0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    nT = len(Ts)

    sig_bsm=0.20
    sig_m=0.15; lam_m=0.8; mJm=-0.08; sJm=0.12

    V0=0.04; kap=2.; th=0.04; xi=0.3; rho=-0.7
    liS=0.5; liV=0.3; bS=5.; bV=5.
    eSS=0.8; eVV=0.6; eSV=0.3; eVS=0.2
    mJd=-0.05; sJd=0.10; mVd=0.02

    alphas=[0.1,0.25,0.4]
    a_lab=[r'$\alpha$=0.1 (H=0.4)',r'$\alpha$=0.25 (H=0.25)',r'$\alpha$=0.4 (H=0.1)']

    # Storage: model -> arrays of length nT
    res = {}  # key: (model, option_type, exercise) -> array

    # ---- BSM ----
    print("[1] BSM ...")
    bsm_ep = np.array([bsm(S0,K_put,T,r,sig_bsm,'put') for T in Ts])
    bsm_ec = np.array([bsm(S0,K_call,T,r,sig_bsm,'call') for T in Ts])
    bsm_ap = np.zeros(nT); bsm_ac = np.zeros(nT)
    for j,T in enumerate(Ts):
        Ns=max(int(Ns_per_year*T),30)
        Sg=gbm_paths(S0,r,sig_bsm,T,Ns,Np)
        bsm_ap[j]=lsm(Sg,K_put,r,T,'put')
        bsm_ac[j]=lsm(Sg,K_call,r,T,'call')
        print(f"  T={T:.2f} done")
    res[('BSM','put','euro')]=bsm_ep; res[('BSM','put','amer')]=bsm_ap
    res[('BSM','call','euro')]=bsm_ec; res[('BSM','call','amer')]=bsm_ac

    # ---- Merton ----
    print("[2] Merton ...")
    mer_ep=np.array([merton_euro(S0,K_put,T,r,sig_m,lam_m,mJm,sJm,'put') for T in Ts])
    mer_ec=np.array([merton_euro(S0,K_call,T,r,sig_m,lam_m,mJm,sJm,'call') for T in Ts])
    mer_ap=np.zeros(nT); mer_ac=np.zeros(nT)
    for j,T in enumerate(Ts):
        Ns=max(int(Ns_per_year*T),30)
        Sm=merton_paths(S0,r,sig_m,lam_m,mJm,sJm,T,Ns,Np)
        mer_ap[j]=lsm(Sm,K_put,r,T,'put')
        mer_ac[j]=lsm(Sm,K_call,r,T,'call')
        print(f"  T={T:.2f} done")
    res[('Merton','put','euro')]=mer_ep; res[('Merton','put','amer')]=mer_ap
    res[('Merton','call','euro')]=mer_ec; res[('Merton','call','amer')]=mer_ac

    # ---- DST ----
    dst_ep={}; dst_ec={}; dst_ap={}; dst_ac={}
    for idx,al in enumerate(alphas):
        print(f"[{3+idx}] DST alpha={al} ...")
        ep=np.zeros(nT); ec=np.zeros(nT)
        ap=np.zeros(nT); ac=np.zeros(nT)
        for j,T in enumerate(Ts):
            Ns=max(int(Ns_per_year*T),30)
            Sd,_=dst_paths(S0,r,V0,kap,th,xi,rho,al,
                           liS,liV,bS,bV,eSS,eVV,eSV,eVS,
                           mJd,sJd,mVd,T,Ns,Np)
            ep[j]=mc_euro(Sd,K_put,r,T,'put')
            ec[j]=mc_euro(Sd,K_call,r,T,'call')
            ap[j]=lsm(Sd,K_put,r,T,'put')
            ac[j]=lsm(Sd,K_call,r,T,'call')
            print(f"  T={T:.2f} done")
        dst_ep[al]=ep; dst_ec[al]=ec; dst_ap[al]=ap; dst_ac[al]=ac

    # ---- CV diagnostics across maturities ----
    print("[CV] Control variate diagnostics ...")
    cv_Np=12000
    cv_Ts=np.array([0.1,0.25,0.5,0.75,1.0,1.5,2.0])
    cv_res={}  # alpha -> {T -> {raw, raw_se, cv, cv_se, bsm_ref}}
    for al in alphas:
        d_al={}
        for T in cv_Ts:
            Ns=max(int(Ns_per_year*T),30)
            Sd,_,Sg=dst_paths(S0,r,V0,kap,th,xi,rho,al,
                              liS,liV,bS,bV,eSS,eVV,eSV,eVS,
                              mJd,sJd,mVd,T,Ns,cv_Np,
                              return_gbm=True,sig_gbm=sig_bsm)
            pay_d=mc_euro_payoffs(Sd,K_put,r,T,'put')
            pay_g=mc_euro_payoffs(Sg,K_put,r,T,'put')
            bsm_ex=bsm(S0,K_put,T,r,sig_bsm,'put')
            raw_m=pay_d.mean(); raw_se=pay_d.std()/np.sqrt(cv_Np)
            cv_m,cv_se=cv_estimate(pay_d,pay_g,bsm_ex)
            d_al[T]=dict(raw=raw_m,raw_se=raw_se,cv=cv_m,cv_se=cv_se,bsm=bsm_ex)
        cv_res[al]=d_al

    # ============================================================
    # PLOTTING  3 × 2
    # ============================================================
    print("Plotting ...")
    c_dst=['#E63946','#457B9D','#2A9D8F']
    c_bsm='#264653'; c_mer='#E9C46A'

    fig,axes=plt.subplots(3,2,figsize=(17,19.5))
    fig.suptitle(
        'Option Price vs Time-to-Maturity: DST (Rough Heston + Dual Hawkes)  |  Merton  |  BSM\n'
        r'$S_0$=100,  K=100 (ATM),  r=3%,  18 000 paths',
        fontsize=15,fontweight='bold',y=0.995)

    def sty(ax,xl,yl,tl,loc='upper left'):
        ax.set_xlabel(xl,fontsize=12); ax.set_ylabel(yl,fontsize=12)
        ax.set_title(tl,fontsize=13,fontweight='bold')
        ax.legend(fontsize=9,loc=loc); ax.grid(True,alpha=0.3)

    # (a) Euro Put vs T
    ax=axes[0,0]
    ax.plot(Ts,bsm_ep,color=c_bsm,lw=2.5,ls='--',marker='D',ms=5,label='BSM')
    ax.plot(Ts,mer_ep,color=c_mer,lw=2.5,marker='s',ms=5,label='Merton JD')
    for i,al in enumerate(alphas):
        ax.plot(Ts,dst_ep[al],color=c_dst[i],lw=2,marker='o',ms=4,label=f'DST {a_lab[i]}')
    sty(ax,'Time to Maturity T (years)','Put Price','(a) European Put Price (ATM, K=100)')

    # (b) Euro Call vs T
    ax=axes[0,1]
    ax.plot(Ts,bsm_ec,color=c_bsm,lw=2.5,ls='--',marker='D',ms=5,label='BSM')
    ax.plot(Ts,mer_ec,color=c_mer,lw=2.5,marker='s',ms=5,label='Merton JD')
    for i,al in enumerate(alphas):
        ax.plot(Ts,dst_ec[al],color=c_dst[i],lw=2,marker='o',ms=4,label=f'DST {a_lab[i]}')
    sty(ax,'Time to Maturity T (years)','Call Price','(b) European Call Price (ATM, K=100)')

    # (c) Amer Put vs T
    ax=axes[1,0]
    ax.plot(Ts,bsm_ap,color=c_bsm,lw=2.5,ls='--',marker='D',ms=5,label='BSM')
    ax.plot(Ts,mer_ap,color=c_mer,lw=2.5,marker='s',ms=5,label='Merton JD')
    for i,al in enumerate(alphas):
        ax.plot(Ts,dst_ap[al],color=c_dst[i],lw=2,marker='o',ms=4,label=f'DST {a_lab[i]}')
    sty(ax,'Time to Maturity T (years)','Put Price','(c) American Put Price (ATM, K=100, LSM)')

    # (d) Amer Call vs T
    ax=axes[1,1]
    ax.plot(Ts,bsm_ac,color=c_bsm,lw=2.5,ls='--',marker='D',ms=5,label='BSM')
    ax.plot(Ts,mer_ac,color=c_mer,lw=2.5,marker='s',ms=5,label='Merton JD')
    for i,al in enumerate(alphas):
        ax.plot(Ts,dst_ac[al],color=c_dst[i],lw=2,marker='o',ms=4,label=f'DST {a_lab[i]}')
    sty(ax,'Time to Maturity T (years)','Call Price','(d) American Call Price (ATM, K=100, LSM)')

    # (e) Early exercise premium Put
    ax=axes[2,0]
    ax.plot(Ts,bsm_ap-bsm_ep,color=c_bsm,lw=2.5,ls='--',marker='D',ms=5,label='BSM')
    ax.plot(Ts,mer_ap-mer_ep,color=c_mer,lw=2.5,marker='s',ms=5,label='Merton JD')
    for i,al in enumerate(alphas):
        ax.plot(Ts,dst_ap[al]-dst_ep[al],color=c_dst[i],lw=2,marker='o',ms=4,label=f'DST {a_lab[i]}')
    ax.axhline(0,color='grey',lw=.8)
    sty(ax,'Time to Maturity T (years)','Price Difference',
        '(e) Early Exercise Premium vs T: Amer − Euro Put')

    # (f) CV diagnostics: VR ratio vs T and SE comparison
    ax=axes[2,1]
    # Two sub-axes via twinx: bars for VR ratio, lines for SE
    bar_w=0.06
    x_base=cv_Ts
    for i,al in enumerate(alphas):
        vr=[]; raw_ses=[]; cv_ses=[]
        for T in cv_Ts:
            d=cv_res[al][T]
            vr.append((d['raw_se']/d['cv_se'])**2 if d['cv_se']>1e-12 else 1.)
            raw_ses.append(d['raw_se']); cv_ses.append(d['cv_se'])
        offset=(i-1)*bar_w
        ax.bar(x_base+offset,vr,width=bar_w,color=c_dst[i],alpha=0.75,
               label=f'VR ratio — {a_lab[i]}')
    ax.axhline(1,color='grey',lw=1.2,ls='--',label='No improvement (=1)')
    ax.set_xlabel('Time to Maturity T (years)',fontsize=12)
    ax.set_ylabel('Variance Reduction Ratio $(\\sigma_{raw}/\\sigma_{CV})^2$',fontsize=12)
    ax.set_title('(f) Control Variate Efficiency vs Maturity (ATM Put)',
                 fontsize=13,fontweight='bold')
    ax.legend(fontsize=8.5,loc='upper left',ncol=2)
    ax.grid(True,alpha=0.3,axis='y')
    ax.set_xticks(cv_Ts)
    ax.set_xticklabels([f'{t:.2f}' for t in cv_Ts],fontsize=9)

    # Add SE annotation on right axis
    ax2=ax.twinx()
    for i,al in enumerate(alphas):
        raw_ses=[cv_res[al][T]['raw_se'] for T in cv_Ts]
        cv_ses=[cv_res[al][T]['cv_se'] for T in cv_Ts]
        ax2.plot(cv_Ts,raw_ses,color=c_dst[i],ls=':',lw=1.5,marker='^',ms=4,alpha=0.7)
        ax2.plot(cv_Ts,cv_ses,color=c_dst[i],ls='-',lw=1.5,marker='v',ms=4,alpha=0.7)
    # Dummy handles for legend
    ax2.plot([],[],color='grey',ls=':',lw=1.5,marker='^',ms=4,label='Raw SE')
    ax2.plot([],[],color='grey',ls='-',lw=1.5,marker='v',ms=4,label='CV SE')
    ax2.set_ylabel('Standard Error',fontsize=11)
    ax2.legend(fontsize=9,loc='upper right')

    plt.tight_layout(rect=[0,0,1,0.965])
    out='/home/claude/option_price_vs_maturity.png'
    plt.savefig(out,dpi=180,bbox_inches='tight',facecolor='white',edgecolor='none')
    plt.close()

    # ---- Print tables ----
    print("\n"+"="*110)
    print("Control Variate Diagnostics — ATM European Put vs Maturity")
    print("="*110)
    for al in alphas:
        print(f"\n  alpha = {al}  (H = {0.5-al:.2f})")
        print(f"  {'T':>6} {'Raw':>10} {'Raw SE':>10} {'CV':>10} {'CV SE':>10} {'VR':>8} {'BSM ref':>10}")
        print(f"  {'-'*72}")
        for T in cv_Ts:
            d=cv_res[al][T]
            vr=(d['raw_se']/d['cv_se'])**2 if d['cv_se']>1e-12 else 0
            print(f"  {T:>6.2f} {d['raw']:>10.4f} {d['raw_se']:>10.4f} "
                  f"{d['cv']:>10.4f} {d['cv_se']:>10.4f} {vr:>7.2f}x {d['bsm']:>10.4f}")

    print("\n"+"="*110)
    print(f"{'Model':<28} | {'T=0.25':>8} {'T=0.50':>8} {'T=1.00':>8} {'T=2.00':>8}"
          f" | {'T=0.25':>8} {'T=0.50':>8} {'T=1.00':>8} {'T=2.00':>8}")
    print(f"{'':<28} | {'--- European Put ---':^35} | {'--- American Put ---':^35}")
    print("-"*110)
    sel=[0.25,0.5,1.0,2.0]; si=[np.argmin(np.abs(Ts-t)) for t in sel]
    def row(name,ep,ap):
        print(f"{name:<28} | {ep[si[0]]:>8.4f} {ep[si[1]]:>8.4f} {ep[si[2]]:>8.4f} {ep[si[3]]:>8.4f}"
              f" | {ap[si[0]]:>8.4f} {ap[si[1]]:>8.4f} {ap[si[2]]:>8.4f} {ap[si[3]]:>8.4f}")
    row('BSM',bsm_ep,bsm_ap)
    row('Merton JD',mer_ep,mer_ap)
    for i,al in enumerate(alphas):
        row(f'DST α={al} (H={0.5-al:.2f})',dst_ep[al],dst_ap[al])
    print("="*110)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Saved to {out}")

if __name__=='__main__':
    main()
