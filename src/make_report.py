"""Build a five-content-page English WORKING report plus references.

External evidence is never invented. Finalization requires verified evidence.json
and actual completed personal checks; otherwise every page is marked WORKING.
"""
import html, json
from pathlib import Path
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def read(name): return json.loads((R/name).read_text())

def build():
    for name,file in [('Body','Arial.ttf'),('Bold','Arial Bold.ttf')]:
        p=Path('/System/Library/Fonts/Supplemental')/file
        if not p.exists(): raise RuntimeError('Set font paths to Unicode-capable regular/bold TTF on this OS.')
        pdfmetrics.registerFont(TTFont(name,str(p)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='B',fontName='Body',fontSize=9,leading=12,spaceAfter=6))
    styles.add(ParagraphStyle(name='S',fontName='Body',fontSize=7.6,leading=9.5,spaceAfter=4))
    styles.add(ParagraphStyle(name='H',fontName='Bold',fontSize=11,leading=14,spaceBefore=6,spaceAfter=6))
    styles.add(ParagraphStyle(name='T',fontName='Bold',fontSize=16,leading=19,spaceAfter=9,alignment=TA_CENTER))
    s=[]
    def para(text,style='B'): return Paragraph(text,styles[style])
    def add(text,style='B'): s.append(para(text,style))
    def table(rows,widths,small=True,padding=3):
        t=Table([[para(html.escape(str(c)), 'S' if small else 'B') for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#55758c')),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#cbd5dc')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding)]))
        s.append(t); s.append(Spacer(1,6))
    m=pd.read_csv(R/'metrics.csv'); cohort=read('cohort.json'); env=read('environment.json'); log=read('log_audit.json'); ab=read('content_ablation.json'); examples=read('two_users.json')
    checks=json.loads((ROOT/'personal_checks.json').read_text())
    evidence=json.loads((ROOT/'evidence.json').read_text()) if (ROOT/'evidence.json').exists() else {}
    complete=all(checks[k].get('status')=='received_and_correct' for k in ['ap5','interval']) and checks['reflection'].get('status')=='received' and checks.get('trust_boundary',{}).get('status')=='received' and evidence.get('verified') is True
    output=ROOT.parent/'final/uminsky_exam_report.pdf' if complete else ROOT/'report_WORKING.pdf'
    output.parent.mkdir(parents=True,exist_ok=True)
    state='FINAL' if complete else 'WORKING — required evidence pending'
    add('Habitual Viewing versus Content and Collaborative Signals on SWEET.TV','T')
    add('Alexander Uminsky · Individual<br/>aleksandr.uminski@gmail.com · 1 November 2026<br/>FKN final exam · '+state,'S')
    add('Abstract','H')
    add('Predicting television preferences requires distinguishing programme consumption from channel tuning in a largely unlabelled schedule. We independently implement content profiles, directional co-view rules and weighted implicit matrix factorisation, compare them with popularity and repeat baselines on one chronological hold-out, and reconstruct qualified viewing through interval unions. Repeat achieves MAP@5 of 0.355864 over all 1,260 template users, compared with 0.246649 for popularity. For this frequency-based offline task, preserving habitual item counts is more effective than the tested semantic and collaborative models, while discovery and cold-start quality remain limited.')
    add('1. Introduction and related work','H')
    add('SWEET.TV requires five ranked programme IDs for each test account. A person may stay on one channel across several broadcasts or watch one show in fragments, so a raw log row cannot directly serve as an item interaction. For example, the real 1,800-second broadcast used in the student check has three viewing fragments, whose union determines its label. The contributions are an overlap-correct reconstruction, a shared temporal comparison of three required methods, and an evidenced failure investigation.')
    add('Kaggle defines the screen-time threshold and top-frequency targets [1]. Hu, Koren and Volinsky separate implicit preference from observation confidence and derive weighted ALS [2, §4]; our implementation uses their log-confidence option with epsilon 1. Sparse categorical item vectors follow DictVectorizer documentation [3, §8.2.1]. Directional pair confidence follows the support/confidence definitions in the mlxtend maintainer guide [4]; we implement pair counting directly and add shrinkage. These sources were opened beyond abstracts. PageRank was considered for diffusion across a co-view graph, but not implemented or measured. A hybrid was tested only after its components.')
    add('2. Method','H')
    add('Join half-open viewing and broadcast intervals on the same exact channel; retain every positive overlap, including earlier-starting and concurrent airings. Sort schedule starts and use a prefix maximum of ends to bound each search. Union fragments per (user, exact airing), then qualify when union seconds / full programme duration ≥ 0.8 [1]. Count one event per qualified airing. A fixed seconds cut would treat short and long programmes inconsistently and would not match the published construction. Session duration is not programme duration.')
    add('All IDs remain strings, including leading zeros. Drop ID 0, missing/invalid IDs or channels, invalid dates, and nonpositive intervals. Blank metadata gives no token. Deduplicate identical (channel, start, end, item) airings; retain genuinely overlapping airings. Compute viewing length from stop–start rather than the misspelled duraton column. A label indicates recorded consumption, not enjoyment. The schedule has only about 11% labelled rows; omitted airtime is not negative feedback.')
    add('Features: category, three genre fields, year string, director and comma-split actors; training-item IDF and L2-normalized vectors. A normalized log-count-weighted user profile scores cosine similarity. Rules use each user’s entire training history as a basket: n(i,j)/(n(i)+λ), no self-rules, at most 100 strongest consequents, weighted by the user’s L1-normalized log counts. ALS uses p=1[count&gt;0], c=1+α log(1+count), unobserved c=1, regularization 1 and eight alternating updates. Only training-observed items participate in ALS fitting. Popularity counts distinct training users; repeat scores personal airing counts.')
    s.append(PageBreak())
    add('3. Experiments and verification','H')
    add('Training contains complete airings in [9 Mar, 18 May 2020); validation contains complete airings in [18 May, 27 Jul). Airings crossing the boundary are excluded. Viewing fragments are clipped to eligible airings. File names do not define windows: schedules spill into adjacent weeks. Vocabulary, IDF and representative model metadata use completed training airings only. Future schedules establish availability, not content features or behaviour. Candidates are the 6,376 labelled IDs airing in validation. Repeats are allowed for every method. Rank by positive score, then training distinct-user popularity, then lexical exact ID; nonpositive scores share the popularity fallback. Cold users get the same fallback.')
    add(f'The log has {log["rows"]:,} rows and {cohort["all_users"]:,} accounts. Train/validation contain {cohort["train_qualified_airings"]:,}/{cohort["val_qualified_airings"]:,} qualifying user-airing observations and {cohort["train_user_item_pairs"]:,}/{cohort["val_user_item_pairs"]:,} user-item pairs. Primary evaluation includes every template user: 1,260 accounts, 18 without training history and 22 without validation targets (AP=0). Sensitivity uses the 1,238 target-active template users; all-active evaluation additionally covers 4,643 accounts in results/metrics.csv. There are 1,372 validation-watched IDs unseen in training interactions. The template is a provided selected population, not a random platform sample.')
    add('Relevance is each user’s five most frequent qualifying validation IDs, frequency ties lexical; fewer than five remain if necessary. AP@5 = sum over first relevant hits of (hits so far / rank), divided by min(5, target count); empty targets score zero. MAP is the macro mean. Prediction order matters; ground-truth membership is treated as a set. This is a local approximation to Kaggle labels, whose private raw construction cannot be inspected.')
    rows=[['Configuration','MAP@5: all 1,260','Active 1,238','Fit + rank, s']]
    for x in m.itertuples(): rows.append([x.method,f'{x.map5_template_all:.6f}',f'{x.map5_template_active:.6f}',f'{x.seconds:.3f}'])
    table(rows,[180,115,105,110])
    add(f'Eight configurations were declared before measurement (search_plan.json): rules (minimum co-users, shrinkage)=(2,5),(5,20); ALS (factors,α)=(16,10),(32,10),(32,40), seed 20261101. No private score was available for tuning. Selected repeat was refit through 26 July and filtered to the 12-week future schedule [27 Jul,19 Oct). Its 1,260-row CSV has five distinct available nonzero IDs in exact template order; four accounts still lack qualified history. Clean final pipeline: {env["wall_seconds"]:.2f}s, peak RSS {env["peak_rss_bytes"]/2**30:.2f} GiB, macOS 27.2 arm64, 10 logical CPUs, Python 3.13.9, BLAS limited to one thread. NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1; full resolved versions are saved. Timings exclude installation and report rendering. The first measured build is preserved separately.')
    add('Failure investigation and confirmation','H')
    add(f'Content underperformed popularity and gave the heavy account only news, AP=0.05, despite four children’s IDs in its held-out Top-5. Metadata similarity omits an explicit item-frequency term: the identity and habitual strength of a show can be lost in the profile. A declared one-variable test changed β from 0 to 0.25 in cosine + β·log(1+personal count)/max-user-logcount. MAP rose from {ab["map5_before"]:.6f} to {ab["map5_after"]:.6f}; 727 users improved and 44 worsened. Heavy-account AP rose to 0.333333. This supports that diagnosis, but does not isolate every metadata effect; pure repeat still wins. The additional hybrid was not adopted.')
    add('Six synthetic tests passed: independent brute-force overlap (positive, nested and multiple matches), union/double counting, date/ID/boundary handling, independent AP arithmetic, tie fallback, and ALS invariance to extra unobserved item columns. An independent indicator-vector AP implementation confirmed every method metric and direct frequency sorting confirmed every final CSV row. Review found availability-only ALS factors influencing its initial Gram matrix; fitting only training-observed columns fixed this, and affected results were rerun. The analysis script also exposed and corrected a genuine integer-versus-string target lookup error; the original traceback is retained. These are assistant checks, separate from student arithmetic.','S')
    s.append(PageBreak())
    add('4. Two-account comparison and strategy','H')
    heavy,cold=examples
    add(f'We select the target-active template account with the most training qualified airings (heavy: {heavy["user_id"]}) and the one with the least, breaking ties by exact user ID (cold: {cold["user_id"]}). This deterministic contrast is the same for all configurations. It is illustrative, not an unbiased sample; selection uses target activity only to ensure a held-out comparison exists. Heavy has {heavy["training_qualified_airings"]} qualified airings across {heavy["training_distinct_items"]} items; cold has zero.')
    counts=heavy['history_all']; hids=[x['id'] for x in heavy['history'][:6]]
    add('Heavy’s six most frequent training IDs (count): '+', '.join(f'{x} ({counts[x]})' for x in hids)+'. Titles/categories are in the dictionary on page 4. Cold has no qualified training items. Full recorded qualified histories and targets are saved in results/two_users.json; no absence is interpreted as never seen.')
    add('Heavy held-out Top-5: '+', '.join(x['id'] for x in heavy['heldout_targets'])+'.<br/>Cold held-out Top-5: '+', '.join(x['id'] for x in cold['heldout_targets'])+'.')
    rows=[['Heavy account: method','Rank 1','Rank 2','Rank 3','Rank 4','Rank 5']]
    for name,rec in heavy['recommendations'].items(): rows.append([name]+[x['id'] for x in rec])
    rows.append(['content + repeat β=.25']+ab['two_users'][0]['after'])
    table(rows,[135,75,75,75,75,75])
    add('Each ranked ID maps to an exact schedule title and category on page 4. For the cold account, <b>all eight configurations and the diagnostic hybrid</b> return the identical ordered list below because every method uses the same no-history fallback; there is no personalised signal to distinguish them.')
    table([['Cold: all methods','Rank 1','Rank 2','Rank 3','Rank 4','Rank 5'],['Common fallback']+[x['id'] for x in cold['recommendations']['popularity']]],[135,75,75,75,75,75])
    add('Heavy’s repeat list recalls four of its five targets (AP=0.76), mainly children’s programmes plus news. Content recalls only the news target at rank 4 (AP=0.05). Rules concentrate on common news and obtain AP=0.20; their directional conditional counts do not encode which show the account repeatedly consumes most. The best ALS configuration (16 factors, α=10) has no hit for this account; its latent representation shares information across co-view patterns but the tested binary-preference objective does not directly rank future top-frequency counts. The 32-factor α=10 list does include Kung Fu Panda, showing that one anecdote cannot select the model.')
    add('For cold, all methods hit two news targets at ranks 1–2 (AP=0.40), yet miss the account’s other documentary, series and children’s targets. This is a concrete fallback limitation. The single fallback list is heavily concentrated in news, and it cannot infer an unseen account’s mix of preferences.')
    add('I select repeat with popularity fallback for this task: it provides the highest measured MAP, inexpensive fitting and transparent explanations from personal counts. Across the template, repeat recommends 275 distinct IDs and averages only 0.167 items absent from qualified training history per five slots; it is strong at habit continuation but weak at discovery. Rules reach only 12 IDs and their most exposed ID appears for 99.84% of users, versus 75.48% for repeat. Content averages 2.132 items absent from that history but only 0.033 corresponding held-out target hits per account: unfamiliarity alone is not useful discovery. ALS-16 has 0.140 such hits and rules 0.169, compared with 0.025 for repeat. These are offline target matches, not evidence of enjoyment, first-ever discovery, engagement or revenue.')
    add('Content attributes can connect programmes even without shared viewers, but our conservative temporal protocol cannot use future-only metadata. Association gives explicit one-hop directional evidence; factorisation compresses collective patterns and may generalize beyond explicit pairs. PageRank could spread a personalised seed over multiple co-view paths and potentially reach different items; no PageRank result is claimed. A next validation experiment would test a time-decayed repeat count to address changing habits, with one declared decay parameter and unchanged cohort/candidates.')
    s.append(PageBreak())
    add('Programme dictionary for both accounts','H')
    add('Exact strings below are representative schedule rows: choose the first validation-window start for each ID (stable airing order). The same ID can label multiple episode titles; these rows identify the recommendation’s programme family, not a predicted episode. No display title or future metadata enters fitting. Category strings retain the original Russian labels. Every ID used in the ordered examples and held-out targets is included.','S')
    dictionary={x['id']:x for user in examples for key in ['heldout_targets','history'] for x in user[key]}
    dictionary.update({x['id']:x for user in examples for rec in user['recommendations'].values() for x in rec})
    used={x['id'] for user in examples for rec in user['recommendations'].values() for x in rec}|{x['id'] for user in examples for x in user['heldout_targets']}|set(hids)|set(ab['two_users'][0]['after'])
    missing=used-set(dictionary)
    if missing:
        data=ROOT.parent/'data/extracted' if (ROOT.parent/'data/extracted').exists() else ROOT/'data/extracted'
        rows=[]
        for filename in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv']:
            for d in pd.read_csv(data/filename,dtype=str,keep_default_na=False,chunksize=150000):
                d=d[d.tv_show_id.isin(missing)].copy()
                d['date']=pd.to_datetime(d.start_time,format='%d.%m.%Y %H:%M:%S')
                rows.append(d[(d.date>=pd.Timestamp('2020-05-18'))&(d.date<pd.Timestamp('2020-07-27'))])
        d=pd.concat(rows).sort_values('date').drop_duplicates('tv_show_id')
        for x in d.itertuples(): dictionary[x.tv_show_id]=dict(id=x.tv_show_id,title=x.tv_show_title,category=x.tv_show_category)
    table([['Exact ID','Exact schedule title','Category']]+[[x,dictionary[x]['title'],dictionary[x]['category']] for x in sorted(used)],[65,365,80],padding=1)
    if evidence.get('screenshot_verified'):
        add('Observed Kaggle submission result','H')
        add('Original student-supplied screenshot, visually inspected: submission.csv, Complete (after deadline), private score '+html.escape(str(evidence['private_score']))+'. Public score '+html.escape(str(evidence['public_score']))+' is distinct. The screenshot is embedded unchanged.','S')
        im=Image(evidence['screenshot_path']); im._restrictSize(480,298); s.append(im)
    s.append(PageBreak())
    add('5. Discussion, personal verification and external result','H')
    add(f'Limits start with missing item labels: {958622:,} of {1074235:,} early schedule rows have ID 0. The log also contains 117 duraton mismatches, so timestamp differences are authoritative. Union reduced raw matched overlap from {log["raw_overlap_seconds"]:,} to {log["union_overlap_seconds"]:,} seconds. Long sessions may represent an unattended television. Concurrent airings may create ambiguous labels; our explicit multi-match policy preserves them. Only one chronological split was used, and its ten-week validation differs from the 12-week Kaggle horizon; no uncertainty interval or online experiment was conducted. Neither the hidden labels nor their exact implementation can be verified locally.')
    add('Student verification and reflection','H')
    add('The student independently checked the real repeat prediction for account 10008352999832730058: hits at ranks 1 and 3, precision 1 and 2/3, sum 5/3, divided by five targets gives AP@5=1/3≈0.3333. The answer was supplied before being shown the computed answer and matches the independent implementation. His verbatim Russian response and assistant check are preserved in personal_checks.json and the native recording.')
    if checks['interval'].get('status')=='received_and_correct':
        add('He also checked the real 1,800-second airing: fragment lengths 662, 102 and 676 seconds sum to a union of 1,440 seconds, giving 0.8; the ≥0.8 threshold includes equality. This example has disjoint fragments and does not establish personal verification of overlapping unions.')
    else: add('WORKING: the personal interval calculation has not yet been recorded as checked.')
    if checks['reflection'].get('status')=='received':
        add(html.escape(checks['reflection']['english_translation']))
    else: add('WORKING: the student’s own explanation of the normalization choice, observed limitation and trusted/unverified quantities is pending. No assistant-written reflection is attributed to him.')
    if checks.get('trust_boundary',{}).get('status')=='received':
        add(html.escape(checks['trust_boundary']['english_translation']))
    else: add('WORKING: an explicit student statement naming the quantities trusted without personal verification is pending. No blanket verification is attributed to him.','S')
    add('Repository and genuine competition evidence','H')
    if evidence.get('screenshot_verified'):
        add('Repository: <link href="'+html.escape(evidence['repository_url'])+'">'+html.escape(evidence['repository_url'])+'</link>. Returned private score: '+html.escape(str(evidence['private_score']))+' (screenshot on page 4). Submission reference: '+html.escape(str(evidence.get('submission_reference') or 'not yet supplied; not visible in screenshot'))+'.','S')
        add('Project, public-bundle and downloaded GitHub CSV bytes match: SHA-256 '+html.escape(evidence['csv_sha256'])+'. The student states that project/submission.csv was uploaded without resaving; the Kaggle-uploaded bytes were not independently downloaded. GitHub was read at commit '+html.escape(evidence['repository_commit'][:12])+'. '+('Implementation folders were verified.' if evidence.get('repository_complete') else 'WORKING: only root files are currently uploaded; implementation folders are missing.'),'S')
    else:
        add('WORKING: repository URL, COMPLETE submission reference, genuine private-score screenshot and CSV identity confirmation are not yet provided. No private score is claimed. This placeholder is text, not a simulated Kaggle screenshot. The designated CSV SHA-256 is de641c8068e7d3f0bc93719a074a3b8ca1055ecb3b4836369e02635248af17e5. Manual browser instructions accompany the public bundle.','S')
    add('6. AI-use disclosure','H')
    add('A previous attempt, including a solution, was made with ChatGPT Work, but lacked a verified native session export. ChatGPT Work prepared this handoff, instructions, prompts and recording utilities; the student did not author those materials. No earlier code, predictions, report text or measured results were imported. This report describes a fresh implementation and measurements in one Codex CLI 0.166.0 session, selected model gpt-6.1-sol as exposed by native turn metadata. Codex read requirements, obtained source descriptions, generated code and English prose, ran tests and experiments, diagnosed failures and prepared artifacts. The student directed the work and supplied the personal checks identified above. Assistant tests are not student tests.')
    add('The native record covers this new session only, not the earlier ChatGPT conversation. Whether the prior assistance also requires an additional log must be clarified with the instructor; no exception or grading acceptance is asserted. The bound session is 01a196eb-7cde-7743-a95b-23446e2f52fc. Structural checks and exact probe-output inspection passed before experiments. The unmodified native JSONL must be copied only after the session ends, kept private and submitted separately; the recording checker does not certify the academic result.','S')
    s.append(PageBreak())
    add('References','H')
    for ref in json.loads((ROOT/'references.json').read_text()):
        add(f'[{ref["number"]}] {html.escape(ref["author"])}, “{html.escape(ref["title"])},” '+str(ref.get('year','online documentation'))+'. Available: <link href="'+html.escape(ref['url'])+'">'+html.escape(ref['url'])+'</link>. Accessed 2026-11-01.')
        add('Sections consulted: '+html.escape(ref['opened_sections'])+'.','S')
    def footer(c,doc):
        c.setFont('Body',7); c.setFillColor(colors.HexColor('#596976'))
        c.drawString(42,25,'Alexander Uminsky · SWEET.TV · '+('FINAL' if complete else 'WORKING'))
        c.drawRightString(553,25,str(doc.page))
    doc=SimpleDocTemplate(str(output),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=40,title='Habitual Viewing versus Content and Collaborative Signals on SWEET.TV',author='Alexander Uminsky')
    doc.build(s,onFirstPage=footer,onLaterPages=footer)
    print(output)

if __name__=='__main__': build()
