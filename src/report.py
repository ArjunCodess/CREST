"""Generate inspectable figures/tables and a paper evidence fragment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def escape(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def build(run,output,fixture=False):
    run=Path(run); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    marker='Software fixture only' if fixture else 'Public training holdout'
    cv=pd.read_csv(run/'temporal_cv.csv')
    cv.to_csv(output/'all_validation_results.csv',index=False)
    if 'selected' in cv:
        selected=cv[cv.selected.fillna(True).astype(bool)].copy()
    else: selected=cv.copy()
    selected.to_csv(output/'baseline_validation.csv',index=False)
    refinement=run/'refinement_cv.csv'
    if refinement.exists():
        refined=pd.read_csv(refinement)
        # A winning comparison without its completed full refit is still pending.
        available=refined.panel.map(lambda p: (run/f'{p}_refined.pkl').exists())
        refined=refined[~refined.accepted | available].copy()
        cv=pd.concat([cv,refined],ignore_index=True)
        for panel in refined.loc[refined.accepted,'panel'].unique():
            if not (run/f'{panel}_refined.pkl').exists(): continue
            replacement=refined[(refined.panel==panel)&(refined.model=='residual')&(refined.projection=='fd')].copy()
            replacement['selected']=True
            selected=pd.concat([selected[selected.panel!=panel],replacement],ignore_index=True)
    selected.to_csv(output/'temporal_cv.csv',index=False)
    if 'model' in cv:
        comparable=cv.dropna(subset=['model','projection']).copy()
        panel_means=comparable.groupby(['model','projection','panel'],as_index=False).S_state.mean()
        panel_means['family']=panel_means.panel.str.rsplit('_',n=1).str[0]
        ablations=panel_means.groupby(['model','projection','family'],as_index=False).S_state.mean().groupby(['model','projection'],as_index=False).S_state.mean()
        counts=panel_means.groupby(['model','projection'],as_index=False).panel.nunique().rename(columns={'panel':'n_panels'})
        ablations=ablations.merge(counts,on=['model','projection'])
        ablations.to_csv(output/'ablations.csv',index=False)
        lines=[r'\begin{table}[t]',r'\caption{Family-macro artificial-mask holdout scores. Panel coverage varies, so partial-panel scores are not directly comparable with ten-panel scores. Model selection uses this same validation set.}',
               r'\centering\begin{tabular}{llrr}',r'\hline Model & FD map & Panels & $S_{state}$ \\',r'\hline']
        for r in ablations.itertuples(): lines.append(f'{escape(r.model)} & {r.projection} & {r.n_panels} & {r.S_state:.4f} \\\\')
        lines.extend([r'\hline\end{tabular}',r'\end{table}'])
        (output/'ablations_table.tex').write_text('\n'.join(lines))
    cv=selected
    if len(cv):
        by_panel=cv.groupby('panel',as_index=False).S_state.mean()
        by_panel['family']=by_panel.panel.str.rsplit('_',n=1).str[0]
        score=float(by_panel.groupby('family').S_state.mean().mean())
    else: score=None
    rows=[]
    for r in cv.itertuples():
        rows.append(f'{escape(r.panel)} & {r.regime} & {r.rmse_speed_kmh:.2f} & {r.rmse_flow_vph_per_lane:.2f} & {r.S_state:.3f} \\\\')
    table='\n'.join([r'\begin{table}[t]',r'\caption{'+escape(marker)+r': artificial-mask temporal holdout. These are not leaderboard scores.}',
        r'\centering\scriptsize',r'\begin{tabular}{llrrr}',r'\hline Panel & Regime & Speed RMSE & Flow/lane & $S_{state}$ \\',r'\hline',
        *rows,r'\hline',r'\end{tabular}',r'\end{table}'])
    (output/'results_table.tex').write_text(table,encoding='utf-8')
    files=sorted(run.glob('*_states.npz'))
    if not files: raise FileNotFoundError('No reconstructed state arrays; run predict first')
    preferred=run/'D12_I5_N_validation_states.npz'
    state_file=preferred if preferred.exists() else files[0]
    bundle=np.load(state_file); states=bundle['prediction']; v=states[...,0]; q=states[...,1]; k=q/np.maximum(v,1)
    rng=np.random.default_rng(42); ix=rng.choice(v.size,min(12000,v.size),replace=False)
    fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
    sc=ax.scatter(k.ravel()[ix],q.ravel()[ix],c=v.ravel()[ix],s=5,alpha=.6,cmap='viridis')
    ax.set(xlabel='Derived density (vehicles/km, all lanes)',ylabel='Flow (vehicles/hour)',title='Reconstructed states: '+state_file.stem.replace('_states',''))
    fig.colorbar(sc,ax=ax,label='Speed (km/h)'); fig.savefig(output/'fundamental_diagram.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
    width=min(len(v),288)
    means=v.mean(axis=1); rolling=np.convolve(means,np.ones(width)/width,mode='valid')
    start=int(np.argmin(rolling))
    im=ax.imshow(v[start:start+width].T,aspect='auto',origin='lower',cmap='RdYlGn',vmin=0,vmax=120)
    ax.set(xlabel='Five-minute step within selected interval',ylabel='Link index (ID order)',title='Lowest-speed 24-hour reconstructed interval')
    fig.colorbar(im,ax=ax,label='Speed (km/h)'); fig.savefig(output/'space_time.png',dpi=180); plt.close(fig)
    availability={'data_kind':'integration_fixture' if fixture else 'official_public_release',
                  'temporal_family_macro_S_state':score,'task1':'Artificial-mask purged holdout available',
                  'task2':'Official labels withheld; official score unavailable',
                  'task3':'Official boundary fluxes withheld; official score unavailable',
                  'task4':'S_link computable; S_od/S_dev/S_attr unavailable',
                  'overall_official_score':None}
    availability['figure_source']=state_file.name
    availability['heatmap_interval_start']=str(bundle['times'][start])
    availability['selected_models']=(selected[['panel','model']].drop_duplicates().to_dict('records') if 'model' in selected else [])
    (output/'evidence.json').write_text(json.dumps(availability,indent=2))
    return availability

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fixture',action='store_true'); a=ap.parse_args(); build(a.run,a.output,a.fixture)
