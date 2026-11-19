"""Offline smoke and resource probe for the generated project; never trains or refreshes."""
import argparse
import json
import os
import platform
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-linux',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.require_linux and platform.system()!='Linux':
        raise SystemExit('Linux execution required; this host cannot verify Linux installation.')
    os.environ['BRENT_ALLOW_PROVIDER_ACCESS']='0'
    start=time.perf_counter();cpu=time.process_time()
    from web.vercel_runtime import app
    import torch
    if torch.__version__!='2.15.0+cpu':raise RuntimeError('Frozen PyTorch version required')
    checks=[]
    def call(endpoint,query=''):
        state=[];stamp=time.perf_counter()
        body=b''.join(app({'REQUEST_METHOD':'GET','PATH_INFO':'/api/forecast','QUERY_STRING':'__brent_endpoint='+endpoint+('&'+query if query else '')},lambda s,h:state.append(s)))
        checks.append({'endpoint':endpoint,'query':query,'status':state[0],'seconds':time.perf_counter()-stamp,'response_bytes':len(body)})
        if not state[0].startswith('200'):raise RuntimeError('Verified resources unavailable: '+endpoint)
        return json.loads(body)
    call('ready');cold=time.perf_counter()-start
    call('overview','period=MAX');call('comparison','horizon=252&metric=r2');call('evaluation')
    for model in ['RNN','LSTM','GRU','persistence']:
        for horizon in [5,63,252]:call('predict',f'model={model}&horizon={horizon}')
    report={'platform':platform.platform(),'torch':torch.__version__,'cold_seconds':cold,
        'cpu_seconds':time.process_time()-cpu,'checks':checks,'providers_enabled':False,
        'note':'Local WSGI execution, not a Vercel deployment, quota or cold-container guarantee.'}
    if platform.system()=='Linux':
        import resource
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if report['peak_rss_bytes']>=2_000_000_000:raise RuntimeError('Hobby memory ceiling exceeded')
    elif platform.system()=='Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in [
                'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise RuntimeError('Windows memory measurement failed')
        report['peak_working_set_bytes']=counters.PeakWorkingSetSize
        report['peak_commit_bytes']=counters.PeakPagefileUsage
    if cold>=240:raise RuntimeError('Insufficient margin below 300-second function duration')
    if max(c['response_bytes'] for c in checks)>=4_500_000:raise RuntimeError('Function response too large')
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print('PASS: verified CPU resources, API requests and resource probe; see report for platform.')


if __name__=='__main__':main()
