"""Execute plain Python notebook cells sequentially and capture their real stdout.

This small executor is deliberately explicit; these cells use no magics or widgets.
The accompanying analyze.py is also runnable independently.
"""
import contextlib, io, os, sys
from pathlib import Path
import nbformat

root=Path(__file__).resolve().parents[1]
data=root.parent/'data/extracted' if (root.parent/'data/extracted').exists() else root/'data/extracted'
cells=[nbformat.v4.new_markdown_cell('# SWEET.TV — executed validation analysis\nFresh Codex implementation by Alexander Uminsky, 2026-11-01. No private test labels.\n\nExecuted by `src/make_notebook.py`, a sequential Python-cell executor with actual stdout capture. See `AI_USE.md`. Synthetic protocol checks are separate from these measured results.'),
nbformat.v4.new_code_cell('from pathlib import Path\nimport json, sys\nimport pandas as pd\nROOT=Path.cwd()\nprint(pd.read_csv(ROOT/"results/metrics.csv").to_string(index=False))\nprint("Selected:", json.loads((ROOT/"results/selected.json").read_text())["method"])'),
nbformat.v4.new_code_cell('sys.path.insert(0,str(ROOT/"src"))\nfrom analyze import run\nfrom threadpoolctl import threadpool_limits\nDATA=ROOT.parent/"data/extracted" if (ROOT.parent/"data/extracted").exists() else ROOT/"data/extracted"\nwith threadpool_limits(limits=1):\n    run(DATA)'),
nbformat.v4.new_code_cell('users=json.loads((ROOT/"results/two_users.json").read_text())\nfor user in users:\n    print("User",user["user_id"],"training airings",user["training_qualified_airings"])\n    print("History:", [(x["id"],x["title"],x["category"]) for x in user["history"]])\n    print("Held-out targets:",user["heldout_targets"])\n    for method,items in user["recommendations"].items():\n        print(method,[(x["id"],x["title"],x["category"]) for x in items])'),
nbformat.v4.new_code_cell('print("Final contract:", (ROOT/"results/submission_contract.json").read_text())\nprint("Independent checks:",(ROOT/"results/independent_submission_check.json").read_text())\nprint("Trust boundary: the assistant ran these calculations; student arithmetic and external Kaggle evidence are recorded separately.")')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3.13','language':'python','name':'python3'}})
namespace={'__name__':'__notebook__'}
os.chdir(root)
count=0
for cell in nb.cells:
    if cell.cell_type!='code': continue
    count+=1; buf=io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(cell.source,f'analysis.ipynb:cell{count}','exec'),namespace)
    cell.execution_count=count
    cell.outputs=[nbformat.v4.new_output(output_type='stream',name='stdout',text=buf.getvalue())]
nbformat.validate(nb)
nbformat.write(nb,root/'analysis.ipynb')
print('Executed',count,'cells; wrote',root/'analysis.ipynb')
