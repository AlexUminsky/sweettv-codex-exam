"""Copy an explicit allowlist into github_upload; never include native journals/data."""
import hashlib, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'github_upload'
DEST.mkdir(exist_ok=True)
files=[]
for name in ['README.md','AI_USE.md','MANUAL_UPLOAD_RU.md','REQUIREMENT_STATUS.md','requirements.txt','search_plan.json','diagnostic_plan.json','references.json','personal_checks.json','analysis.ipynb','submission.csv']:
    files.append(Path(name))
for folder in ['src','tests']:
    files.extend(p.relative_to(ROOT) for p in (ROOT/folder).glob('*.py'))
for name in ['metrics.csv','selected.json','environment.json','review_cached_environment.json','analysis_environment.json','cohort.json','two_users.json','content_ablation.json','discovery_diagnostics.csv','independent_checks.json','independent_submission_check.json','submission_contract.json','log_audit.json','schedule_audit.json']:
    files.append(Path('results')/name)
files.extend(p.relative_to(ROOT) for p in (ROOT/'results/validation_predictions').glob('*.csv'))
files.extend(p.relative_to(ROOT) for p in (ROOT/'results/initial_run').iterdir() if p.suffix in ['.csv','.json'])
files.extend(p.relative_to(ROOT) for p in (ROOT/'logs').glob('*.log'))
for name in ['input_manifest.json','archive_check.json','kaggle_pages.json','kaggle_data_description.txt','report_visual_review.json']:
    if (ROOT/'audit'/name).exists(): files.append(Path('audit')/name)
rows=[]
for relative in files:
    source=ROOT/relative
    assert source.is_file(), str(source)
    assert source.suffix not in ['.jsonl','.zip']
    target=DEST/relative; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==hashlib.sha256(target.read_bytes()).hexdigest()
    rows.append(dict(path=str(relative),bytes=target.stat().st_size,sha256=digest))
(DEST/'.gitignore').write_text('.venv/\n__pycache__/\ndata/\n*.jsonl\nreport_pages/\n')
(DEST/'BUNDLE_MANIFEST.json').write_text(json.dumps(rows,indent=2)+'\n')
for path in DEST.rglob('*'):
    assert path.suffix not in ['.jsonl','.zip','.npz','.gz'],str(path)
    assert not set(path.relative_to(DEST).parts)&{'.venv','data','sessions','archived_sessions','pdf_reader'},str(path)
    assert path.name not in ['auth.json','config.toml'],str(path)
assert (DEST/'submission.csv').read_bytes()==(ROOT/'submission.csv').read_bytes()
print('Public bundle verified:',len(rows),'files,',sum(x['bytes'] for x in rows),'bytes; CSV byte-identical.')
