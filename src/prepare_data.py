"""CRC-check and safely extract the five exact input CSVs; no account access."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path

EXPECTED={'dataset11-30.csv','export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv','submission.csv'}

def prepare(archive, destination, manifest):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==5 and set(z.namelist())==EXPECTED
        assert z.testzip() is None, 'ZIP CRC check failed'
        rows=[]
        for info in z.infolist():
            assert Path(info.filename).name==info.filename
            target=destination/info.filename
            expected_hash=hashlib.sha256()
            with z.open(info) as source, target.open('wb') as out:
                for block in iter(lambda:source.read(1<<20),b''):
                    expected_hash.update(block); out.write(block)
            actual=hashlib.sha256(target.read_bytes()).hexdigest()
            assert target.stat().st_size==info.file_size and actual==expected_hash.hexdigest()
            rows.append(dict(name=info.filename,bytes=info.file_size,crc32=info.CRC,sha256=actual))
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    print('CRC, five safe paths, sizes and SHA-256 verified.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('archive',type=Path); p.add_argument('destination',type=Path); p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'audit/input_manifest.json'); a=p.parse_args()
    prepare(a.archive,a.destination,a.manifest)
