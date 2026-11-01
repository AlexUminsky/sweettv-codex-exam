"""New SWEET.TV implementation, authored in the recorded Codex session.

Run from any directory: python project/src/pipeline.py --data data/extracted
Timestamps are treated as naive dataset wall times; no undocumented timezone conversion.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CUT = '2020-05-18 00:00:00'
END = '2020-07-27 00:00:00'
START = '2020-03-09 00:00:00'
TEST_END = '2020-10-19 00:00:00'
SEED = 20261101

def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n')

def seconds(values, fmt):
    x = pd.to_datetime(values, format=fmt, errors='coerce')
    return x.to_numpy(dtype='datetime64[s]').astype('int64')

def epoch(value):
    return int(np.datetime64(value, 's').astype('int64'))

def interval_pairs(starts, ends, session_starts, session_ends):
    """All positive overlaps, including earlier and overlapping broadcasts.

    Starts must be sorted. Prefix maximum of ends makes the lower bound safe even
    for nested airings. Returned indices refer to sessions and airings.
    """
    if not len(starts) or not len(session_starts):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lo = np.searchsorted(np.maximum.accumulate(ends), session_starts, side='right')
    hi = np.searchsorted(starts, session_ends, side='left')
    counts = np.maximum(hi-lo, 0)
    si = np.repeat(np.arange(len(counts)), counts)
    offsets = np.repeat(np.cumsum(counts)-counts, counts)
    ai = np.repeat(lo, counts)+np.arange(counts.sum())-offsets
    valid = ends[ai] > session_starts[si]
    return si[valid], ai[valid]

def union_fragments(frame):
    """Union by exact user-airing; fragments from different chunks coalesce too."""
    frame = frame.sort_values(['u', 'a', 's', 'e'], kind='stable').copy()
    previous = frame.groupby(['u','a'], sort=False).e.cummax().groupby(
        [frame.u, frame.a], sort=False).shift()
    frame['seconds'] = np.maximum(0, frame.e-np.maximum(frame.s, previous.fillna(frame.s)))
    return frame.groupby(['u','a'], sort=False).seconds.sum().reset_index()

def ap5(predicted, targets):
    targets = set(targets)
    if not targets:
        return 0.0
    seen = set(); hits = 0; score = 0.0
    for rank, item in enumerate(predicted[:5], 1):
        if item in targets and item not in seen:
            hits += 1; score += hits/rank
        seen.add(item)
    return score/min(5,len(targets))

def top5(scores, eligible, popularity):
    """Positive scores first; zero/invalid scores share popularity fallback.

    eligible is lexicographically sorted exact string IDs represented by indices.
    Primary score, then training popularity, then lexical ID; same rule everywhere.
    """
    s = np.nan_to_num(scores[eligible], nan=0., posinf=0., neginf=0.)
    s = np.maximum(s, 0.)
    order = np.lexsort((eligible, -popularity[eligible], -s))
    return eligible[order[:5]]

def read_schedule(data):
    parts=[]; audit=[]
    for name in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
        rows=0; zero=0; invalid=0; lo=None; hi=None
        for d in pd.read_csv(data/name, dtype=str, keep_default_na=False, chunksize=150000):
            rows+=len(d); zero+=int((d.tv_show_id=='0').sum())
            s=seconds(d.start_time, '%d.%m.%Y %H:%M:%S')
            dur=pd.to_numeric(d.duration,errors='coerce').to_numpy(dtype=float)
            valid=(s>0)&np.isfinite(dur)&(dur>0)&d.channel_id.ne('').to_numpy()&d.tv_show_id.str.fullmatch(r'\d+').to_numpy()
            invalid+=int((~valid).sum())
            if valid.any():
                lo=int(s[valid].min()) if lo is None else min(lo,int(s[valid].min()))
                hi=int(s[valid].max()) if hi is None else max(hi,int(s[valid].max()))
            d=d.loc[valid & d.tv_show_id.ne('0').to_numpy()].copy()
            # Preserve only labelled rows in memory; ID 0 is not negative feedback.
            d['s']=seconds(d.start_time,'%d.%m.%Y %H:%M:%S')
            d['duration_s']=pd.to_numeric(d.duration).astype('int64')
            d['e']=d.s+d.duration_s
            parts.append(d)
        audit.append(dict(file=name,rows=rows,zero_rows=zero,invalid_rows=invalid,min_start=lo,max_start=hi))
    d=pd.concat(parts,ignore_index=True)
    before=len(d)
    d=d.drop_duplicates(['channel_id','s','e','tv_show_id']).sort_values(['channel_id','s','e','tv_show_id']).reset_index(drop=True)
    d['a']=np.arange(len(d))
    dump(ROOT/'results/schedule_audit.json',dict(files=audit,duplicate_airings_removed=before-len(d),labelled_airings=len(d),duration_quantiles=d.duration_s.quantile([0,.1,.5,.9,.99,1]).to_dict()))
    return d

def join_logs(data, schedule):
    timetable={}
    # Test airings are never used to join views or build model features.
    for c,g in schedule[(schedule.s>=epoch(START))&(schedule.e<=epoch(END))].groupby('channel_id',sort=False):
        timetable[c]=(g.s.to_numpy(),g.e.to_numpy(),g.a.to_numpy())
    fragments=[]; users=set(); log_audit=collections.Counter(); durations=[]; first=None; last=None
    for chunk_no,d in enumerate(pd.read_csv(data/'dataset11-30.csv',dtype=str,keep_default_na=False,chunksize=200000)):
        log_audit['rows']+=len(d)
        s=seconds(d.start_time,'%Y-%m-%d %H:%M:%S'); e=seconds(d.stop_time,'%Y-%m-%d %H:%M:%S')
        dur=pd.to_numeric(d.duraton,errors='coerce').to_numpy(dtype=float)
        valid=(s>0)&(e>s)&d.user_id.ne('').to_numpy()&d.vsetv_id.ne('').to_numpy()
        log_audit['invalid_rows']+=int((~valid).sum())
        log_audit['duration_mismatch_rows']+=int((valid & (dur != e-s)).sum())
        log_audit['sessions_crossing_validation_boundary']+=int(((s<epoch(CUT))&(e>epoch(CUT))&valid).sum())
        d=d.loc[valid].copy(); d['s']=s[valid]; d['e']=e[valid]
        users.update(d.user_id)
        durations.append((e-s)[valid])
        first=int(s[valid].min()) if first is None else min(first,int(s[valid].min()))
        last=int(e[valid].max()) if last is None else max(last,int(e[valid].max()))
        for c,g in d.groupby('vsetv_id',sort=False):
            if c not in timetable:
                log_audit['sessions_on_channels_without_labelled_airings']+=len(g); continue
            st,en,ids=timetable[c]
            si,ai=interval_pairs(st,en,g.s.to_numpy(),g.e.to_numpy())
            if len(ai):
                fragments.append(pd.DataFrame({'u':g.user_id.to_numpy()[si], 'a':ids[ai], 's':np.maximum(g.s.to_numpy()[si],st[ai]), 'e':np.minimum(g.e.to_numpy()[si],en[ai])}))
        if chunk_no%5==0: print('joined log rows',log_audit['rows'],flush=True)
    raw=pd.concat(fragments,ignore_index=True)
    union=union_fragments(raw)
    union=union.merge(schedule[['a','tv_show_id','s','e','duration_s']],on='a',validate='many_to_one')
    assert (union.seconds<=union.duration_s).all()
    log_audit['overlap_fragments']=len(raw); log_audit['user_airing_unions']=len(union)
    log_audit['raw_overlap_seconds']=int((raw.e-raw.s).sum()); log_audit['union_overlap_seconds']=int(union.seconds.sum())
    log_audit['first_session_start']=first; log_audit['last_session_end']=last
    log_audit['session_duration_quantiles']={str(q):float(v) for q,v in zip([0,.1,.5,.9,.99,1],np.quantile(np.concatenate(durations),[0,.1,.5,.9,.99,1]))}
    dump(ROOT/'results/log_audit.json',dict(log_audit))
    union.to_csv(ROOT/'results/user_airing_overlap.csv.gz',index=False)
    # Real threshold exercise: minimal fragment count, fraction close to .8.
    sizes=raw.groupby(['u','a']).size().rename('fragments')
    examples=union.join(sizes,on=['u','a'])
    examples['distance']=(examples.seconds/examples.duration_s-.8).abs()
    ex=examples[(examples.fragments>=2)&(examples.fragments<=4)].sort_values(['distance','u','a']).iloc[0]
    rr=raw[(raw.u==ex.u)&(raw.a==ex.a)]
    air=schedule.loc[schedule.a==ex.a].iloc[0]
    dump(ROOT/'results/interval_exercise_private.json',dict(user_id=ex.u,airing_id=int(ex.a),item_id=air.tv_show_id,title=air.tv_show_title,air_start=int(air.s),air_end=int(air.e),duration=int(ex.duration_s),fragments=rr.to_dict('records'),union_seconds=int(ex.seconds)))
    return union,users

def interaction_matrix(union, users, items, start, end, threshold=.8):
    w=union[(union.s>=epoch(start))&(union.e<=epoch(end))&(union.seconds>=threshold*union.duration_s)]
    uid={u:i for i,u in enumerate(users)}; iid={v:i for i,v in enumerate(items)}
    counts=w.groupby(['u','tv_show_id'],sort=False).size()
    mat=sparse.coo_matrix((counts.to_numpy(dtype=np.float64),([uid[u] for u,i in counts.index],[iid[i] for u,i in counts.index])),shape=(len(users),len(items))).tocsr()
    return mat,w

def item_features(schedule, items, end):
    # Only completed training airings can supply metadata. Latest row is deterministic.
    known=schedule[(schedule.s>=epoch(START))&(schedule.e<=epoch(end))].sort_values(['s','a']).drop_duplicates('tv_show_id',keep='last').set_index('tv_show_id')
    docs=[]
    for item in items:
        doc={}
        if item in known.index:
            row=known.loc[item]
            for key in ['tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director']:
                val=str(row[key]).strip().lower()
                if val: doc[key+'='+val]=1.
            for actor in str(row.actors).split(','):
                if actor.strip(): doc['actor='+actor.strip().lower()]=1.
        docs.append(doc)
    # Fit vocabulary and inverse document frequency on training-known IDs only.
    train_idx=[i for i,x in enumerate(items) if x in known.index]
    vectorizer=DictVectorizer(sparse=True)
    vectorizer.fit([docs[i] for i in train_idx])
    x=vectorizer.transform(docs).tocsr()
    df=np.asarray((x[train_idx]>0).sum(axis=0)).ravel()
    idf=np.log((1+len(train_idx))/(1+df))+1
    x=normalize(x.multiply(idf).tocsr())
    return x,dict(training_metadata_items=len(train_idx),vocabulary=len(vectorizer.feature_names_))

def fit_als(counts, factors=32, alpha=10., reg=1., iterations=8):
    """Hu--Koren--Volinsky weighted implicit ALS; unobserved c=1, p=0.

    Observed c=1+alpha*log1p(number of >=80% watched airings).
    Each update uses full fixed-side Gram plus observed-only correction.
    """
    # Compact fitting space contains training-observed items only. Schedule-only
    # IDs cannot change initialization, the Gram matrices, or trained parameters.
    total_items=counts.shape[1]
    observed=np.flatnonzero(np.asarray(counts.sum(axis=0)).ravel()>0)
    counts=counts[:,observed]
    rng=np.random.default_rng(SEED)
    u=np.zeros((counts.shape[0],factors))
    v=rng.normal(0,.01,(counts.shape[1],factors))
    trans=counts.T.tocsr(); eye=np.eye(factors)*reg
    def update(target, fixed, matrix):
        gram=fixed.T@fixed+eye
        for i in range(matrix.shape[0]):
            a,b=matrix.indptr[i:i+2]; idx=matrix.indices[a:b]
            if not len(idx): target[i]=0; continue
            c=alpha*np.log1p(matrix.data[a:b]); y=fixed[idx]
            lhs=gram+(y.T*c)@y; rhs=y.T@(1+c)
            target[i]=np.linalg.solve(lhs,rhs)
    for _ in range(iterations):
        update(u,v,counts); update(v,u,trans)
    full_v=np.zeros((total_items,factors)); full_v[observed]=v
    return u,full_v

def predictions(counts,schedule,items,train_end,future_start,future_end,configs):
    n_users,n_items=counts.shape
    eligible_ids=set(schedule.loc[(schedule.s>=epoch(future_start))&(schedule.s<epoch(future_end)),'tv_show_id'])
    eligible=np.array([i for i,x in enumerate(items) if x in eligible_ids])
    assert len(eligible)>=5
    popularity=np.asarray((counts>0).sum(axis=0)).ravel().astype(float)
    binary=counts.copy(); binary.data[:]=1.
    weighted=counts.copy(); weighted.data=np.log1p(weighted.data)
    result={}; timings={}; extra={}
    for cfg in configs:
        name=cfg['name']; begin=time.perf_counter(); kind=cfg['kind']
        if kind=='popularity': scores=np.broadcast_to(popularity,(n_users,n_items))
        elif kind=='repeat': scores=counts.toarray()
        elif kind=='content':
            features, info=item_features(schedule,items,train_end); extra['content']=info
            profiles=normalize(weighted@features)
            scores=(profiles@features.T).toarray()
        elif kind=='rules':
            co=(binary.T@binary).tocoo()
            support=popularity
            keep=(co.row!=co.col)&(co.data>=cfg['min_support'])
            # Directional confidence P(j | i), shrunk by support(i)+lambda.
            rules=sparse.coo_matrix((co.data[keep]/(support[co.row[keep]]+cfg['shrink']), (co.row[keep],co.col[keep])),shape=(n_items,n_items)).tocsr()
            # Retain strongest 100 consequents per antecedent, deterministic ties.
            for i in range(n_items):
                a,b=rules.indptr[i:i+2]
                if b-a>100:
                    rank=np.lexsort((rules.indices[a:b],-rules.data[a:b])); rules.data[a+rank[100:]]=0
            rules.eliminate_zeros()
            scores=(normalize(weighted,norm='l1')@rules).toarray()
            extra[name]={'rules':rules.nnz,'formula':'n_ij/(n_i+shrink), i != j; >=min_support co-view users'}
        elif kind=='als':
            u,v=fit_als(counts,**{k:v for k,v in cfg.items() if k not in ['name','kind']})
            scores=u@v.T
        else: raise ValueError(kind)
        cold=np.asarray(counts.sum(axis=1)).ravel()==0
        # A user's missing collaborative history always invokes identical fallback.
        scores=np.array(scores,copy=True); scores[cold,:]=0
        pred=np.vstack([top5(row,eligible,popularity) for row in scores])
        result[name]=pred
        timings[name]=time.perf_counter()-begin
        print('model',name,'seconds',round(timings[name],3),flush=True)
    extra['candidate_count']=len(eligible)
    extra['candidate_ids']=sorted(eligible_ids)
    return result,timings,extra

def targets_from(counts,items):
    out=[]
    for row in counts:
        idx=row.indices; values=row.data
        order=np.lexsort((idx,-values))[:5]
        out.append([items[i] for i in idx[order]])
    return out

def save_predictions(name,pred,users,items,folder):
    pd.DataFrame({'user_id':users,'tv_show_id':[' '.join(items[i] for i in row) for row in pred]}).to_csv(folder/(name+'.csv'),index=False)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True); parser.add_argument('--reuse-overlap',action='store_true'); args=parser.parse_args()
    begin=time.perf_counter(); (ROOT/'results/validation_predictions').mkdir(parents=True,exist_ok=True)
    np.random.seed(SEED)
    print('start',pd.Timestamp.now().isoformat(),flush=True)
    configs=json.loads((ROOT/'search_plan.json').read_text())['configs']
    schedule=read_schedule(args.data)
    template=pd.read_csv(args.data/'submission.csv',dtype=str,keep_default_na=False)
    if args.reuse_overlap:
        union=pd.read_csv(ROOT/'results/user_airing_overlap.csv.gz',dtype={'u':str,'tv_show_id':str})
        users=set(pd.read_csv(ROOT/'results/all_users.csv',dtype=str).user_id)
    else:
        union,users=join_logs(args.data,schedule)
        pd.DataFrame({'user_id':sorted(users)}).to_csv(ROOT/'results/all_users.csv',index=False)
    users=sorted(users|set(template.user_id)); items=sorted(set(schedule.tv_show_id))
    dump(ROOT/'results/index.json',{'users':users,'items':items})
    train,train_w=interaction_matrix(union,users,items,START,CUT)
    val,val_w=interaction_matrix(union,users,items,CUT,END)
    assert (train_w.e<=epoch(CUT)).all() and (val_w.s>=epoch(CUT)).all()
    sparse.save_npz(ROOT/'results/train_counts.npz',train); sparse.save_npz(ROOT/'results/validation_counts.npz',val)
    user_index={u:i for i,u in enumerate(users)}; cohort=np.array([user_index[u] for u in template.user_id])
    targets=targets_from(val,items)
    dump(ROOT/'results/validation_targets.json',dict(zip(users,targets)))
    pred,timings,extra=predictions(train,schedule,items,CUT,CUT,END,configs)
    metrics=[]
    history=np.asarray(train.sum(axis=1)).ravel(); val_activity=np.asarray(val.sum(axis=1)).ravel()
    for name,p in pred.items():
        save_predictions(name,p,users,items,ROOT/'results/validation_predictions')
        aps=np.array([ap5([items[i] for i in row],targets[u]) for u,row in enumerate(p)])
        metrics.append(dict(method=name,map5_template_all=float(aps[cohort].mean()),map5_template_active=float(aps[cohort[val_activity[cohort]>0]].mean()),map5_all_active=float(aps[val_activity>0].mean()),seconds=timings[name]))
    table=pd.DataFrame(metrics); table.to_csv(ROOT/'results/metrics.csv',index=False)
    print(table.to_string(index=False),flush=True)
    winner=table.sort_values(['map5_template_all','method'],ascending=[False,True]).iloc[0].method
    dump(ROOT/'results/selected.json',{'method':winner,'config':next(c for c in configs if c['name']==winner),'selection':'maximum MAP@5 on all 1260 template users, ties lexical method name','metrics':metrics})
    unseen=set(val_w.tv_show_id)-set(train_w.tv_show_id)
    dump(ROOT/'results/cohort.json',dict(all_users=len(users),template_users=len(cohort),template_no_train=int((history[cohort]==0).sum()),template_no_val=int((val_activity[cohort]==0).sum()),template_active_val=int((val_activity[cohort]>0).sum()),all_active_val=int((val_activity>0).sum()),all_no_train=int((history==0).sum()),train_user_item_pairs=train.nnz,val_user_item_pairs=val.nnz,train_qualified_airings=len(train_w),val_qualified_airings=len(val_w),val_items_unseen_train_interactions=len(unseen),validation_candidates=extra['candidate_count'],model_details=extra))
    # Same heavy and lowest-history active template users for every component.
    active=cohort[val_activity[cohort]>0]
    chosen=[int(sorted(active,key=lambda u:(-history[u],users[u]))[0]),int(sorted(active,key=lambda u:(history[u],users[u]))[0])]
    display=schedule[(schedule.s>=epoch(CUT))&(schedule.s<epoch(END))].sort_values(['s','a']).drop_duplicates('tv_show_id').set_index('tv_show_id')
    def describe(ids):
        return [dict(id=x,title=str(display.loc[x].tv_show_title) if x in display.index else str(schedule.loc[schedule.tv_show_id==x].iloc[0].tv_show_title),category=str(display.loc[x].tv_show_category) if x in display.index else str(schedule.loc[schedule.tv_show_id==x].iloc[0].tv_show_category)) for x in ids]
    examples=[]
    for u in chosen:
        row=train[u]; order=np.lexsort((row.indices,-row.data))
        examples.append(dict(user_id=users[u],training_qualified_airings=int(history[u]),training_distinct_items=int(train[u].nnz),history=describe([items[i] for i in row.indices[order[:12]]]),history_all={items[i]:int(v) for i,v in zip(row.indices,row.data)},heldout_targets=describe(targets[u]),recommendations={name:describe([items[i] for i in p[u]]) for name,p in pred.items()}))
    dump(ROOT/'results/two_users.json',examples)
    # Nontrivial real AP exercise, deterministic first user with 1--4 hits.
    pp=pred[winner]
    for u in sorted(cohort,key=lambda i:users[i]):
        ids=[items[i] for i in pp[u]]; hits=sum(x in targets[u] for x in ids)
        if 0<hits<5 and any(x not in targets[u] for x in ids[:3]):
            dump(ROOT/'results/ap_exercise_private.json',dict(user_id=users[u],method=winner,predictions=ids,targets=targets[u],ap5=ap5(ids,targets[u]))); break
    all_train,all_w=interaction_matrix(union,users,items,START,END)
    sparse.save_npz(ROOT/'results/full_counts.npz',all_train)
    # Selection is frozen; refit only the winning method on all allowed weeks.
    final,ft,fe=predictions(all_train,schedule,items,END,END,TEST_END,[next(c for c in configs if c['name']==winner)])
    fp=final[winner][cohort]
    save_predictions('submission',fp,list(template.user_id),items,ROOT)
    eligible=set(fe['candidate_ids'])
    output=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
    assert list(output.columns)==['user_id','tv_show_id'] and list(output.user_id)==list(template.user_id)
    assert len(output)==1260 and output.user_id.is_unique
    for text in output.tv_show_id:
        ids=text.split(); assert len(ids)==len(set(ids))==5 and '0' not in ids and set(ids)<=eligible
    dump(ROOT/'results/submission_contract.json',dict(passed=True,rows=len(output),sha256=hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest(),template_order_preserved=True,five_distinct_nonzero_eligible=True,final_no_history=int((np.asarray(all_train.sum(axis=1)).ravel()[cohort]==0).sum()),refit_seconds=ft))
    dump(ROOT/'results/environment.json',dict(timestamp=pd.Timestamp.now().isoformat(),platform=platform.platform(),machine=platform.machine(),python=sys.version,processor=platform.processor(),cpu_count=os.cpu_count(),seed=SEED,threads=1,wall_seconds=time.perf_counter()-begin,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,validation_boundary=CUT,final_boundary=END,threshold=.8))
    print('complete',winner,'wall seconds',time.perf_counter()-begin,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
