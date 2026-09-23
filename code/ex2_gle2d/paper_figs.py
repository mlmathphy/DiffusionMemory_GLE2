"""Adapt the user's ex14 paper gallery to the validated equilibrium 2D GLE.

python paper_figs.py         # prepare a cached short forecast if needed, plot
python paper_figs.py --plot  # redraw only; never inference or simulation

No training, scout, or long stationary-rollout call. Forecast samples are
illustrative, separate from the unchanged aggregate summary.json results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import config as cfg

COL = {'exact': '#0b0b0b', 'mem': '#2a78d6', 'lstm': '#eb6834', 'markov': '#1baf7a'}
LAB = {'exact': 'exact', 'mem': 'ours', 'lstm': 'LSTM', 'markov': 'memoryless'}
ORDER = ('exact', 'mem', 'lstm', 'markov')
SNAPS = (1, 4, 16)
HORIZON = 64
N_ENS = 3000
SEED = 20260909
PREP_VERSION = 1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def filter_state(refs, trajectory, t):
    """Exact finite-observed-history posterior of the four-state embedding."""
    A, Q, S = refs['Ad'], refs['Qd'], refs['Sigma']
    mean = np.zeros(4)
    mean[:2] = trajectory[0]
    P = S - S[:, :2] @ np.linalg.solve(S[:2, :2], S[:2])
    for j in range(1, t + 1):
        mean = A @ mean
        P = A @ P @ A.T + Q
        K = np.linalg.solve(P[:2, :2], P[:2]).T
        mean += K @ (trajectory[j] - mean[:2])
        P -= K @ P[:2]
        P = .5 * (P + P.T)
    return mean, P


def select_anchor(X, bank_burn, refs):
    """Nearby current velocities, separated Kalman next-step increment means.

    Select using reference quantities only, never a model's error. Among
    the original evaluation anchors with >=1.5 innovation Mahalanobis
    separation, minimize current-velocity separation. If none qualify,
    use the first two anchors and record that the condition was not met.
    Only anchor A is used for the requested showcase gallery.
    """
    from evaluate import kalman_filter_means
    start = max(bank_burn, cfg.EVAL_WARM + 1)
    ts = np.linspace(start, len(X)-2, cfg.N_QUERY).astype(int)
    means = kalman_filter_means(refs, X)[ts] - X[ts]
    inv = np.linalg.inv(refs['V_kal1'])
    candidates = []
    for i in range(len(ts)):
        for j in range(i+1, len(ts)):
            d = means[i] - means[j]
            sep = float(np.sqrt(max(0., d @ inv @ d)))
            if sep >= 1.5:
                candidates.append((float(np.linalg.norm(X[ts[i]]-X[ts[j]])), i, j, sep))
    if candidates:
        dist, i, j, sep = min(candidates)
    else:
        i, j = 0, min(1, len(ts)-1)
        dist = float(np.linalg.norm(X[ts[i]]-X[ts[j]]))
        d = means[i]-means[j]
        sep = float(np.sqrt(max(0., d @ inv @ d)))
    return dict(trajectory=0, times=[int(ts[i]), int(ts[j])],
                criterion_met=bool(candidates), current_distance=dist,
                predictive_mahalanobis=sep,
                rule='Among evaluation-anchor pairs separated by >=1.5 innovation Mahalanobis units in Kalman increment mean, minimize current velocity separation; ties by index. If none qualify use first two. Showcase = first member.')


def prepare(n_ens=N_ENS, horizon=HORIZON, snaps=SNAPS):
    import torch
    from evaluate import load_net, net_rollout
    from train_lstm import LSTMGaussianFull, warm_state_2d
    from conditioning import bank_features
    import exact_refs
    from scout import psd_sqrt

    out = Path(cfg.OUT_DIR)
    needed = ['data.npz', 'compression.npz', 'summary.json'] + [f'{m}/model.pt' for m in ORDER[1:]]
    missing = [str(out/p) for p in needed if not (out/p).exists()]
    if missing:
        raise SystemExit('Use the workstation containing the completed comparison. Missing:\n'+'\n'.join(missing))
    cache = out/'paper_gallery'
    cache.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    digest = hashlib.sha256()
    # Plot-code edits do not invalidate forecasts. Increment PREP_VERSION
    # for changes to preparation semantics in this file.
    inputs = [out/'data.npz', out/'compression.npz']
    folder = Path(__file__).parent
    inputs += [folder/n for n in ('config.py','evaluate.py','conditioning.py',
                                  'exact_refs.py','scout.py','train_lstm.py')]
    inputs += [folder.parent/'memdiff'/'distill.py', folder.parent/'memdiff'/'features.py']
    for p in inputs:
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024), b''): digest.update(block)
    digest.update(str((PREP_VERSION,n_ens,horizon,snaps,SEED,str(device),
                       np.__version__,torch.__version__)).encode())
    base_key = digest.hexdigest()
    refs = exact_refs.references()
    with np.load(out/'data.npz') as data: X = data['X_test'][0]
    with np.load(out/'compression.npz') as c: coef = c['coef']
    rates = refs['rates']
    banks, nb = bank_features(X[None],rates)
    anchor = select_anchor(X, nb, refs)
    t = anchor['times'][0]
    if t >= len(X)-1 or t < cfg.EVAL_WARM: raise ValueError('Insufficient showcase history')
    mean, P = filter_state(refs,X,t)
    v0 = np.repeat(X[t][None],n_ens,axis=0)
    m0 = np.repeat(banks[0,t][None],n_ens,axis=0)
    # Analytical velocity marginals and sampled-displacement MSD, with
    # displacement defined consistently as dt * sum_{j=1}^h v_{n+j}.
    A, Q = refs['Ad'], refs['Qd']
    B = np.zeros((6,6)); B[:4,:4]=A
    B[4:,:4]=cfg.DT*A[:2]; B[4:,4:]=np.eye(2)
    L = np.vstack([np.eye(4), cfg.DT*np.eye(4)[:2]])
    S = np.zeros((6,6)); S[:4,:4]=P
    mm = np.r_[mean, np.zeros(2)]
    exact_mu, exact_cov, msd_x, msd_v = [],[],[],[]
    for h in range(horizon):
        mm=B@mm; S=B@S@B.T+L@Q@L.T
        exact_mu.append(mm[:2].tolist()); exact_cov.append(S[:2,:2].tolist())
        msd_x.append(float(mm[4:]@mm[4:]+np.trace(S[4:,4:])))
        msd_v.append(float(np.sum((mm[:2]-X[t])**2)+np.trace(S[:2,:2])))
    keys = {}
    for k in ORDER:
        d = hashlib.sha256(base_key.encode())
        if k!='exact': d.update((out/k/'model.pt').read_bytes())
        key=d.hexdigest(); keys[k]=key
        path=cache/f'{k}.npz'
        if path.exists():
            with np.load(path) as saved:
                if str(saved['key'])==key:
                    print(f'reuse forecast {k}',flush=True); continue
        print(f'prepare {k}: {n_ens} members x {horizon} steps (checkpoint inference only)',flush=True)
        if k=='exact':
            rng=np.random.default_rng(SEED)
            Y=mean+rng.standard_normal((n_ens,4))@psd_sqrt(P).T
            Y[:,:2]=X[t]
            E=np.empty((n_ens,horizon,2)); sqQ=psd_sqrt(Q)
            for h in range(horizon):
                Y=Y@A.T+rng.standard_normal((n_ens,4))@sqQ.T
                E[:,h]=Y[:,:2]
        elif k in ('mem','markov'):
            net,ck=load_net(k,device)
            E=net_rollout(net,ck,k,coef,rates if k=='mem' else np.array([]),
                          v0,m0 if k=='mem' else np.zeros((n_ens,0)),horizon,SEED+1,device)
        else:
            ck=torch.load(out/'lstm/model.pt',map_location=device,weights_only=False)
            net=LSTMGaussianFull(4,cfg.LSTM_HIDDEN,cfg.LSTM_LAYERS).to(device)
            net.load_state_dict(ck['state']);net.eval(); meta=ck['meta']
            E=np.empty((n_ens,horizon,2))
            # Warm the history ONCE, then expand only recurrent state.
            with torch.no_grad():
                state,v,dv=warm_state_2d(net,meta,X[None,t-cfg.EVAL_WARM:t+1],device)
                state=tuple(s.expand(-1,n_ens,-1).contiguous() for s in state)
                v=v.expand(n_ens,-1).clone();dv=dv.expand(n_ens,-1).clone()
                g=torch.Generator(device=device);g.manual_seed(SEED+1)
                for h in range(horizon):
                    u=torch.cat([v/meta['x_sd'],dv*meta['kappa_s']],-1)[:,None]
                    sample,state=net.sample_step(u,state,generator=g)
                    dv=sample/meta['kappa_s'];v=v+dv;E[:,h]=v.cpu().numpy()
        if not np.isfinite(E).all(): raise ValueError(f'Nonfinite forecast: {k}')
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,key=key,V=E);os.replace(tmp,path)
    meta=dict(version=PREP_VERSION,keys=keys,anchor=anchor,dt=cfg.DT,n_ens=n_ens,
              horizon=horizon,snaps=list(snaps),v0=X[t].tolist(),
              exact_mean=exact_mu,exact_cov=exact_cov,msd_x_exact=msd_x,msd_v_exact=msd_v)
    write_json(cache/'metadata.json',meta)
    return cache


def render(cache=None, fig_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from memdiff.plotting import setup_style
    setup_style()
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
    cache=Path(cache or Path(cfg.OUT_DIR)/'paper_gallery')
    fig_dir=Path(fig_dir or cfg.FIG_DIR);fig_dir.mkdir(parents=True,exist_ok=True)
    if not (cache/'metadata.json').exists():
        raise SystemExit('No forecast cache. Run python paper_figs.py once on the workstation.')
    meta=json.loads((cache/'metadata.json').read_text())
    ens={}
    for k in ORDER:
        with np.load(cache/f'{k}.npz') as d:
            if str(d['key'])!=meta['keys'][k]:raise ValueError(f'Incomplete/mixed forecast cache: {k}; resume preparation')
            ens[k]=d['V']
    snaps=meta['snaps'];dt=meta['dt'];n=meta['n_ens']
    def save(fig,name):
        fig.savefig(fig_dir/name,bbox_inches='tight');plt.close(fig);print(f'figure -> {fig_dir/name}')
    def mass_levels(H):
        a=np.sort(H.ravel())[::-1];cum=np.cumsum(a)/a.sum()
        return np.unique([a[min(np.searchsorted(cum,p),len(a)-1)] for p in (.5,.95)])
    def kde1(values,grid,bw):
        # Bounded intermediate allocations for 3000-member panels.
        y=np.zeros(len(grid))
        for a in range(0,len(values),512):
            y+=np.exp(-.5*((grid[:,None]-values[None,a:a+512])/bw)**2).sum(1)
        return y/(len(values)*bw*np.sqrt(2*np.pi))
    # Common reference-derived frame across every evolution panel.
    mu=np.asarray(meta['exact_mean']);cov=np.asarray(meta['exact_cov'])
    sd=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))
    idx=np.asarray(snaps)-1
    lo=(mu[idx]-4.5*sd[idx]).min(0);hi=(mu[idx]+4.5*sd[idx]).max(0)
    limlo=float(lo.min());limhi=float(hi.max())
    edges=np.linspace(limlo,limhi,121);centers=(edges[1:]+edges[:-1])/2;dx=edges[1]-edges[0]
    fig,axes=plt.subplots(2,len(snaps),figsize=(3*len(snaps),5.4),sharex=True,sharey=True,layout='constrained')
    outside={}
    for c,h in enumerate(snaps):
        ex=ens['exact'][:,h-1];bw=np.sqrt(np.var(ex,axis=0).mean())*n**(-1/6)
        density={}
        for k in ('exact','mem'):
            pts=ens[k][:,h-1];H=np.histogram2d(pts[:,0],pts[:,1],bins=(edges,edges))[0]
            outside[f'{k}_{h}']=float(1-H.sum()/n)
            H=gaussian_filter(H,max(bw/dx,.6));density[k]=H/(max(H.sum(),1)*dx*dx)
        vmax=max(H.max() for H in density.values())
        for r,k in enumerate(('exact','mem')):
            ax=axes[r,c];H=density[k]
            cf=ax.contourf(centers,centers,(H/max(vmax,1e-30)).T,levels=np.linspace(0,1,11),cmap='Blues')
            if H.sum()>0:ax.contour(centers,centers,H.T,levels=mass_levels(H),colors=COL['exact'],linewidths=.7)
            ax.set_aspect('equal');ax.set_xlim(limlo,limhi);ax.set_ylim(limlo,limhi)
            if r==0:ax.set_title(f'$h={h}$, $t={h*dt:g}$')
            if r==1:ax.set_xlabel('$v_x$')
            if c==0:ax.set_ylabel(LAB[k]+'\n$v_y$')
            ax.text(.02,.02,f"{outside[f'{k}_{h}']:.1%} outside",transform=ax.transAxes,fontsize=7)
    fig.colorbar(cf,ax=axes.ravel().tolist(),shrink=.8,label='relative density (normalized per column)')
    save(fig,'ex2_evolution2d.pdf')
    fig,axes=plt.subplots(2,len(snaps),figsize=(3*len(snaps),4.7),layout='constrained')
    for c,h in enumerate(snaps):
        for j in range(2):
            ax=axes[j,c];grid=np.linspace(mu[h-1,j]-4.5*sd[h-1,j],mu[h-1,j]+4.5*sd[h-1,j],240)
            bw=max(ens['exact'][:,h-1,j].std()*n**(-1/5),1e-8)
            for k in ORDER:ax.plot(grid,kde1(ens[k][:,h-1,j],grid,bw),color=COL[k],label=LAB[k],alpha=.85 if k!='exact' else 1)
            ax.set(xlabel=('$v_x$' if j==0 else '$v_y$'),ylabel='density',title=f'$h={h}$, $t={h*dt:g}$')
    axes[0,-1].legend(frameon=False);save(fig,'ex2_evolution1d.pdf')
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained');tt=np.arange(1,meta['horizon']+1)*dt
    for k in ORDER:
        V=ens[k]
        vx=((V-np.asarray(meta['v0']))**2).sum(2).mean(0)
        xx=(np.cumsum(V,axis=1)*dt)**2;xx=xx.sum(2).mean(0)
        if k=='exact':vx=meta['msd_v_exact'];xx=meta['msd_x_exact']
        axes[0].plot(tt,xx,color=COL[k],label=LAB[k]);axes[1].plot(tt,vx,color=COL[k],label=LAB[k])
    axes[0].set(ylabel=r'$\mathbb{E}[|r_{n+h}-r_n|^2\mid H_n]$',title='(a) Conditional sampled-displacement MSD')
    axes[1].set(ylabel=r'$\mathbb{E}[|v_{n+h}-v_n|^2\mid H_n]$',title='(b) Conditional velocity change')
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('forecast time');ax.legend(frameon=False)
    save(fig,'ex2_msd.pdf')
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    for j,ax in enumerate(axes):
        m=mu[0,j]-meta['v0'][j];s=sd[0,j];grid=np.linspace(m-4.5*s,m+4.5*s,240)
        ax.plot(grid,np.exp(-.5*((grid-m)/s)**2)/(s*np.sqrt(2*np.pi)),color=COL['exact'],label='exact Gaussian')
        bw=s*n**(-1/5)
        for k in ORDER[1:]:ax.plot(grid,kde1(ens[k][:,0,j]-meta['v0'][j],grid,bw),color=COL[k],label=LAB[k])
        ax.set(xlabel=('$\\Delta v_x$' if j==0 else '$\\Delta v_y$'),ylabel='density')
    axes[0].legend(frameon=False);save(fig,'ex2_margins.pdf')
    # Aggregate evidence comes verbatim from the completed comparison.
    summary_path=Path(cfg.OUT_DIR)/'summary.json'
    if summary_path.exists():
        s=json.loads(summary_path.read_text());names=[k for k in ORDER[1:] if k in s.get('common_reference',{})]
        fig,ax=plt.subplots(figsize=(5,2.5),layout='constrained');rng=np.random.default_rng(7)
        for i,k in enumerate(names):
            a=np.asarray(s['common_reference'][k]['per_anchor'])
            ax.scatter(np.maximum(a,1e-12),i+rng.uniform(-.15,.15,len(a)),s=8,color=COL[k],alpha=.5)
            ax.vlines(max(float(np.median(a)),1e-12),i-.25,i+.25,color='k')
        ax.set_xscale('log');ax.set_yticks(range(len(names)),[LAB[k] for k in names]);ax.invert_yaxis()
        ax.set_xlabel(r'one-step normalized moment $W_2^2$ vs Kalman (all anchors)')
        save(fig,'ex2_metrics.pdf')
        lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'Model & Covariance error & VACF error & Diffusion error & One-step $W_2^2$ \\',r'\hline']
        for k in names:
            c=s['closed_loop'][k];q=s['common_reference'][k]['median']
            lines.append(f"{LAB[k]} & {c['cov_dev']:.4f} & {c['vacf_err']:.4f} & {c['diff_err']:.4f} & {q:.6f} "+r'\\')
        lines.extend([r'\hline',r'\end{tabular}'])
        (fig_dir/'ex2_table_comparison.tex').write_text('\n'.join(lines)+'\n')
    write_json(fig_dir/'gallery_metadata.json',dict(**meta,outside_evolution2d=outside))
    (fig_dir/'FIGURES.md').write_text(f'''# 2D equilibrium GLE paper gallery

Adapted from the supplied ex14 plotting layout, not its model or data.
Coordinates are the original velocities v_x, v_y; no anisotropic display
rescaling is applied. Ours = estimated-predictive-coordinate EMA memory
plus distilled diffusion. LSTM has a full-covariance Gaussian head.

## Selected paper set
- ex2_evolution2d.pdf: exact (top) and ours (bottom), horizons {snaps} steps
  after the observed conditioning history, physical time h * {dt}.
- ex2_evolution1d.pdf: matching velocity marginals, exact / ours / LSTM /
  memoryless. Both components are shown; limits follow the exact law.
- ex2_msd.pdf: conditional sampled-position MSD r_(n+h)-r_n = dt sum
  v_(n+j), and conditional squared velocity change. Neither is stationary
  MSD. The exact curves are analytic, the model curves use ensembles.
- ex2_table_comparison.tex: existing aggregate comparison values, unchanged.

## Supplementary
- ex2_margins.pdf: one-step increment marginals at the showcase history.
- ex2_metrics.pdf: every stored common-reference evaluation-anchor error;
  bars mark medians, not confidence intervals.
The previous ex2_vacf/accuracy/stationary/conditionals PDFs remain intact.

## Caption disclosures
Evolution and MSD illustrate ONE post-hoc reference-selected history,
trajectory 0 at step {meta['anchor']['times'][0]}, not aggregate evidence.
Selection rule: {meta['anchor']['rule']}
Pair criterion met: {meta['anchor']['criterion_met']}.
Ensemble size {n}. No hidden state is supplied to learned models. The exact
forecast starts from the Kalman posterior conditioned on the observed
record; LSTM uses its declared observed warm-up, mem its full EMA bank.
2D contours show 50/95% mass of the smoothed IN-FRAME estimated density,
not exact Gaussian probability boundaries. Both rows use an identical
exact-derived bandwidth at each horizon. Relative density normalization
is shared within each column; spatial limits are common to all panels.
Outside-frame fractions are shown and stored in gallery_metadata.json.
1D KDE bandwidths also come from the exact ensemble, shared across models.
Finite-ensemble and smoothing differences are not solely model error.
The table uses a single checkpoint per method, no unrecorded error bars.
Network one-step errors use empirical moments; LSTM uses analytic moments.
Diffusion is reported, not a validated success gate. No LSTM-superiority
or failure claim follows from a selected showcase.

## Reuse
python paper_figs.py --plot redraws only. Preparation caches each completed
model forecast separately and reuses matching inputs/checkpoints. No
training or long stationary simulation is part of this command.
''')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    cache=Path(cfg.OUT_DIR)/'paper_gallery'
    if not args.plot:cache=prepare()
    render(cache)


if __name__=='__main__':main()
