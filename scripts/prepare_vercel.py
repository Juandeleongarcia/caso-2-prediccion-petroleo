"""Generate a standalone, code-only Vercel project without touching scientific artifacts."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=ROOT/'deployment/vercel'
FRONTEND=['index.html','style.css','app.mjs','domain.mjs','vendor/plotly.min.js','vendor/LICENSE.plotly.txt']
TEMPLATE_FILES=['api/forecast.py','requirements.txt','vercel.json','.python-version',
                'build.mjs','verify_runtime.py','Dockerfile.check','constraints-linux.txt',
                '.vercelignore','.dockerignore','README.md']


def source_files():
    fixed=['web/__init__.py','web/backend.py','web/vercel_runtime.py',
        'configs/data.json','configs/economic_events.json','configs/serving_bundle.json','configs/dashboard_bundle.json',
        'artifacts/derived/brent-academic-v1/artifact_manifest.json','artifacts/derived/brent-academic-v1/metrics.csv']
    found=set(fixed+['web/frontend/'+p for p in FRONTEND])
    pending=['web/backend.py','web/vercel_runtime.py']
    def add(module):
        if module=='oil_forecast':
            name='src/oil_forecast/__init__.py'
        elif module.startswith('oil_forecast.'):
            name='src/'+module.replace('.','/')+'.py'
        else:return
        if name not in found:
            if not (ROOT/name).is_file():raise ValueError('Unresolved runtime module: '+module)
            found.add(name);pending.append(name)
    add('oil_forecast')
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8'))):
            if isinstance(node,ast.Import):
                for alias in node.names:add(alias.name)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    if not name.startswith('src/oil_forecast/') or node.level!=1:raise ValueError('Unsupported relative import')
                    if node.module:add('oil_forecast.'+node.module)
                    else:
                        for alias in node.names:add('oil_forecast.'+alias.name)
                elif node.module:add(node.module)
    return sorted(found)


def prepare(destination):
    target=Path(destination).absolute()
    allowed=(ROOT/'artifacts/local').resolve()
    if (not target.resolve().is_relative_to(allowed) and target.resolve()!=(ROOT/'deployment/vercel-app').resolve()) or target.exists():
        raise ValueError('Use a new directory within artifacts/local or deployment/vercel-app; existing projects are never overwritten')
    target.parent.mkdir(parents=True,exist_ok=True)
    files=source_files()
    with tempfile.TemporaryDirectory(prefix='.vercel-package-',dir=target.parent) as temporary:
        stage=Path(temporary)/'project';stage.mkdir()
        provenance={}
        for name in files:
            output=stage/name;output.parent.mkdir(parents=True,exist_ok=True)
            output.write_bytes((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))
            provenance[name]=hashlib.sha256(output.read_bytes()).hexdigest()
        for name in TEMPLATE_FILES:
            original=TEMPLATE/name
            output=stage/name;output.parent.mkdir(parents=True,exist_ok=True)
            output.write_bytes(original.read_bytes().replace(b'\r\n',b'\n'))
            provenance[name]=hashlib.sha256(output.read_bytes()).hexdigest()
        (stage/'source_manifest.json').write_text(json.dumps({'kind':'code-only-vercel-project',
            'private_artifacts_included':False,'normalization':'UTF-8 text with LF line endings',
            'files':provenance},indent=2)+'\n',encoding='utf-8')
        stage.rename(target)
    return {'directory':str(target),'files':len(provenance)+1,'bytes':sum(p.stat().st_size for p in target.rglob('*') if p.is_file())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/local/vercel-large-v1')
    print(json.dumps(prepare(parser.parse_args().output),ensure_ascii=False))
