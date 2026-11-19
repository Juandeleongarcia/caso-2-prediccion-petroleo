import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,readdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {apiOrigin,build,publicFiles} from '../frontend/build.mjs';

test('deployment origin rejects absent, insecure and credential-bearing URLs',()=>{
  for(const value of [undefined,'','http://api.example.org','https://user:secret@api.example.org',
    'https://api.example.org/?token=secret','https://api.example.org/#secret','https://api.example.org/api',
    'https://localhost','https://127.0.0.1','https://[::1]']) assert.throws(()=>apiOrigin(value));
  assert.equal(apiOrigin('https://api.example.org/'),'https://api.example.org');
});

test('Vercel output contains only allowlisted public files and public origin',async()=>{
  const root=await mkdtemp(join(tmpdir(),'brent-build-'));
  try {
    const output=join(root,'public');
    assert.deepEqual(await build(output,'https://api.example.org'),[...publicFiles,'config.js']);
    assert.deepEqual((await readdir(output)).sort(),['app.mjs','config.js','domain.mjs','index.html','style.css','vendor']);
    assert.deepEqual((await readdir(join(output,'vendor'))).sort(),['LICENSE.plotly.txt','plotly.min.js']);
    const config=await readFile(join(output,'config.js'),'utf8');
    assert.match(config,/window.BRENT_API_BASE = "https:\/\/api.example.org"/);
    assert.doesNotMatch(config,/HF_TOKEN|checkpoints|Bearer/);
    await assert.rejects(build(output,'https://other.example.org')); // No overwrite.
    assert.equal(await readFile(join(output,'config.js'),'utf8'),config);
  } finally {await rm(root,{recursive:true,force:true});}
});
