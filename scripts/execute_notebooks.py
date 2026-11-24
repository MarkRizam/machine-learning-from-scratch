"""Execute each notebook in an isolated subprocess and save the outputs."""
from pathlib import Path
import json
import subprocess
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'results'/'executed-notebooks'


def execute(path):
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=90,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,TARGET/path.name)


def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)==3 and sys.argv[1]=='--one':
        execute(Path(sys.argv[2]))
        return
    status={}
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        try:
            subprocess.run([sys.executable,__file__,'--one',str(path)],check=True,timeout=150)
            n=nbformat.read(TARGET/path.name,4)
            status[path.name]={'status':'PASS','code_cells_executed':sum(c.cell_type=='code' for c in n.cells),'display_outputs':sum(o.output_type=='display_data' for c in n.cells if c.cell_type=='code' for o in c.outputs)}
        except (subprocess.SubprocessError, OSError) as exc:
            status[path.name]={'status':'FAIL','reason':str(exc)}
        print(path.name,status[path.name]['status'],flush=True)
    (ROOT/'results'/'notebook-status.json').write_text(json.dumps(status,indent=2)+'\n')
    if len(status)!=9 or any(v['status']!='PASS' for v in status.values()):
        raise SystemExit(1)

if __name__=='__main__': main()
