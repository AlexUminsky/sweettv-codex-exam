"""Read-only public repository audit at a pinned commit; no credentials."""
import datetime, hashlib, json, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='https://api.github.com/repos/AlexUminsky/sweettv-codex-exam'

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SWEETTV-local-audit'}),timeout=30).read()

def main():
    data=get(BASE+'/git/trees/main?recursive=1'); tree=json.loads(data)
    assert not tree.get('truncated'), 'Incomplete public tree'
    (ROOT/'audit/github_tree.json').write_bytes(data)
    blobs={x['path']:x for x in tree['tree'] if x['type']=='blob'}
    required=['src/pipeline.py','src/analyze.py','src/prepare_data.py','src/make_notebook.py','tests/test_protocol.py','requirements.txt','README.md','submission.csv','analysis.ipynb','results/metrics.csv']
    comparisons={}
    for path in required:
        if path not in blobs:
            comparisons[path]={'present':False}; continue
        content=(ROOT/path).read_bytes()
        git_sha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        comparisons[path]={'present':True,'matches_local_bytes_via_git_blob_sha1':git_sha==blobs[path]['sha']}
    csv=get('https://raw.githubusercontent.com/AlexUminsky/sweettv-codex-exam/'+tree['sha']+'/submission.csv')
    (ROOT/'audit/github_submission.csv').write_bytes(csv)
    identical=csv==(ROOT/'submission.csv').read_bytes()==(ROOT.parent/'github_upload/submission.csv').read_bytes()
    result=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository='https://github.com/AlexUminsky/sweettv-codex-exam',commit=tree['sha'],remote_blob_count=len(blobs),required_files=comparisons,required_files_present=all(x['present'] for x in comparisons.values()),csv_identical=identical,csv_bytes=len(csv),csv_sha256=hashlib.sha256(csv).hexdigest(),unsafe_public_paths=[x for x in blobs if x.endswith(('.jsonl','.zip')) or any(p in x.split('/') for p in ['.venv','data','sessions','archived_sessions'])])
    (ROOT/'audit/github_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
