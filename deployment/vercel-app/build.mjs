import {mkdir,copyFile,writeFile} from 'node:fs/promises';
if(process.env.VERCEL_SUPPORT_LARGE_FUNCTIONS!=='1')throw Error('Set VERCEL_SUPPORT_LARGE_FUNCTIONS=1 before building.');
await mkdir('public'); // A previous build is never silently overwritten.
await mkdir('public/vendor');
for(const f of ['index.html','style.css','app.mjs','domain.mjs','vendor/plotly.min.js','vendor/LICENSE.plotly.txt'])
  await copyFile(`web/frontend/${f}`,`public/${f}`);
await writeFile('public/config.js',"// Same-origin API; no credentials.\nwindow.BRENT_API_BASE = '';\n");
