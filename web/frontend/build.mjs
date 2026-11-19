// Vercel build: only these public assets can enter the output directory.
import {mkdir, copyFile, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
export const publicFiles = ['index.html','style.css','app.mjs','domain.mjs',
  'vendor/plotly.min.js','vendor/LICENSE.plotly.txt'];

export function apiOrigin(value) {
  const url = new URL(value);
  if(url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/')
    throw Error('BRENT_API_BASE debe ser un origen HTTPS sin credenciales, ruta ni parámetros.');
  if(url.hostname === 'localhost' || !url.hostname.includes('.') || /^[\d.]+$/.test(url.hostname) || url.hostname.startsWith('['))
    throw Error('El backend público necesita un nombre DNS HTTPS.');
  return url.origin;
}

export async function build(output, origin) {
  const api = apiOrigin(origin); // Validate before touching output.
  const source = dirname(fileURLToPath(import.meta.url));
  const target = resolve(output);
  if(target === source || target === resolve(source,'..'))
    throw Error('El destino no puede sustituir el código fuente.');
  await mkdir(target); // Existing destinations are never overwritten or deleted.
  await mkdir(resolve(target,'vendor'));
  for(const file of publicFiles) await copyFile(resolve(source,file),resolve(target,file));
  await writeFile(resolve(target,'config.js'),`// Public API origin; no credentials.\nwindow.BRENT_API_BASE = ${JSON.stringify(api)};\n`);
  return [...publicFiles,'config.js'];
}

if(process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { await build(resolve(dirname(fileURLToPath(import.meta.url)),'public'),process.env.BRENT_API_BASE); }
  catch { console.error('Build detenido: configura BRENT_API_BASE como origen HTTPS y utiliza un destino nuevo.'); process.exitCode=1; }
}
