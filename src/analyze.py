"""Executed analysis and single-variable content ablation. No hidden test labels."""
import collections, contextlib, hashlib, io, json, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import pipeline as p

def independent_ap(pred, target):
    # Independent indicator-vector arithmetic; duplicates do not create hits.
    if not target: return 0.
    indicators=np.array([int(x in target and x not in pred[:i]) for i,x in enumerate(pred[:5])])
    return float(np.sum(np.cumsum(indicators)/np.arange(1,len(indicators)+1)*indicators)/min(5,len(set(target))))

def run(data):
    start=time.perf_counter(); r=p.ROOT/'results'
    ix=json.loads((r/'index.json').read_text()); users=ix['users']; items=ix['items']
    targets=json.loads((r/'validation_targets.json').read_text())
    template=pd.read_csv(data/'submission.csv',dtype=str)
    cohort=np.array([users.index(u) for u in template.user_id])
    train=sparse.load_npz(r/'train_counts.npz')
    population=np.asarray((train>0).sum(axis=0)).ravel()
    schedule=p.read_schedule(data)
    candidate_ids=set(schedule.loc[(schedule.s>=p.epoch(p.CUT))&(schedule.s<p.epoch(p.END)),'tv_show_id'])
    eligible=np.array([i for i,x in enumerate(items) if x in candidate_ids])
    x,info=p.item_features(schedule,items,p.CUT)
    weight=train.copy(); weight.data=np.log1p(weight.data)
    scores=(normalize(weight@x)@x.T).toarray()
    frequency=weight.toarray(); maxima=frequency.max(axis=1); maxima[maxima==0]=1
    frequency/=maxima[:,None]
    pred=np.vstack([p.top5(row,eligible,population) for row in scores+.25*frequency])
    p.save_predictions('content_repeat_lambda025',pred,users,items,r/'validation_predictions')
    baseline=pd.read_csv(r/'validation_predictions/content.csv',dtype=str).set_index('user_id')
    before=np.array([independent_ap(baseline.loc[u].tv_show_id.split(),targets[u]) for u in users])
    after=np.array([independent_ap([items[i] for i in row],targets[users[u]]) for u,row in enumerate(pred)])
    # Count exact sparse metadata collisions among nonzero training-known vectors.
    groups=collections.defaultdict(list)
    for i in eligible:
        row=x[i]
        if row.nnz: groups[(tuple(row.indices),tuple(np.round(row.data,12)))].append(items[i])
    collisions=sorted((v for v in groups.values() if len(v)>1),key=lambda v:(-len(v),v))
    example_users=json.loads((r/'two_users.json').read_text())
    diagnostic={
        'lambda_before':0,'lambda_after':.25,'map5_before':float(before[cohort].mean()),'map5_after':float(after[cohort].mean()),
        'paired_users_improved':int((after[cohort]>before[cohort]).sum()),'paired_users_worsened':int((after[cohort]<before[cohort]).sum()),
        'candidate_nonzero_metadata_vectors':sum(len(v) for v in groups.values()),'identical_vector_groups':len(collisions),'items_in_collisions':sum(map(len,collisions)),
        'largest_collision_ids':collisions[0],
        'two_users':[dict(user_id=e['user_id'],before=[d['id'] for d in e['recommendations']['content']],after=[items[i] for i in pred[users.index(e['user_id'])]],ap_before=before[users.index(e['user_id'])],ap_after=after[users.index(e['user_id'])]) for e in example_users],
        'time_seconds':time.perf_counter()-start,
        'interpretation':'Adding the declared item-specific repeat term tests loss of identity/frequency information; it does not isolate all metadata or population effects.'}
    p.dump(r/'content_ablation.json',diagnostic)
    print('Content diagnostic:',json.dumps(diagnostic,ensure_ascii=False))
    # Independently recalculate all saved original component metrics.
    metric_table=pd.read_csv(r/'metrics.csv'); checks={}
    summary=[]
    for method in metric_table.method:
        d=pd.read_csv(r/'validation_predictions'/f'{method}.csv',dtype=str).set_index('user_id')
        values=[independent_ap(d.loc[u].tv_show_id.split(),targets[u]) for u in template.user_id]
        actual=float(np.mean(values)); expected=float(metric_table.loc[metric_table.method==method,'map5_template_all'].iloc[0])
        assert abs(actual-expected)<1e-12
        # Every validation output respects the same candidate/duplicate contract.
        for text in d.tv_show_id:
            ids=text.split(); assert len(ids)==len(set(ids))==5 and set(ids)<=candidate_ids and '0' not in ids
        checks[method]={'independent_map5':actual,'agreement':True}
        history=[set(items[i] for i in train[users.index(u)].indices) for u in template.user_id]
        lists=[d.loc[u].tv_show_id.split() for u in template.user_id]
        novel=[sum(x not in h for x in ids) for h,ids in zip(history,lists)]
        discovered=[sum(x not in h and x in targets[u] for x in ids) for u,h,ids in zip(template.user_id,history,lists)]
        exposures=collections.Counter(x for ids in lists for x in ids)
        summary.append(dict(method=method,mean_unrecorded_history_recommendations=float(np.mean(novel)),mean_unrecorded_history_target_hits=float(np.mean(discovered)),distinct_recommended_ids=len(exposures),top_item_exposure_share=exposures.most_common(1)[0][1]/1260))
    p.dump(r/'independent_checks.json',checks)
    pd.DataFrame(summary).to_csv(r/'discovery_diagnostics.csv',index=False)
    # Verify final predictions by an independent direct frequency sort.
    full=sparse.load_npz(r/'full_counts.npz'); final_pop=np.asarray((full>0).sum(axis=0)).ravel()
    final_candidates=set(schedule.loc[(schedule.s>=p.epoch(p.END))&(schedule.s<p.epoch(p.TEST_END)),'tv_show_id'])
    submission=pd.read_csv(p.ROOT/'submission.csv',dtype=str)
    for row in submission.itertuples(index=False):
        u=users.index(row.user_id); counts=dict(zip(full[u].indices,full[u].data))
        expected=sorted((i for i,x in enumerate(items) if x in final_candidates),key=lambda i:(-counts.get(i,0),-final_pop[i],items[i]))[:5]
        assert row.tv_show_id.split()==[items[i] for i in expected]
    p.dump(r/'independent_submission_check.json',dict(passed=True,all_1260_rows_recomputed=True,sha256=hashlib.sha256((p.ROOT/'submission.csv').read_bytes()).hexdigest()))
    print('All component MAP calculations and all 1260 final rows independently confirmed.')
    print(pd.DataFrame(summary).to_string(index=False))
    p.dump(r/'analysis_environment.json',dict(wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':
    with threadpool_limits(limits=1): run(Path(sys.argv[1]))
