"""Download unmodified public sources, preserving failures rather than substituting data.
This script never invokes a paid model API and never uploads private text.
"""
from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,hashlib,json
import requests
ROOT=Path(__file__).resolve().parents[1]
URLS=[('french_daily_factors.zip','https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip')]
DATES=['20250129','20250319','20250507','20250618','20250730','20250917','20251029','20251210','20260128','20260318','20260429','20260617','20260729']
URLS += [(f'fomcminutes{x}.html',f'https://www.federalreserve.gov/monetarypolicy/fomcminutes{x}.htm') for x in DATES]
def get(item):
 name,url=item;row={'file':name,'url':url,'attempt_utc':datetime.now(timezone.utc).isoformat()}
 try:
  response=requests.get(url,timeout=(8,20),headers={'User-Agent':'Academic replication source audit/1.0'});response.raise_for_status()
  if name.endswith('.zip') and not response.content.startswith(b'PK'):raise ValueError('Not a ZIP archive')
  if name.endswith('.html') and 'Minutes' not in response.text:raise ValueError('Expected minutes text missing')
  path=ROOT/'external/downloads'/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(response.content)
  row.update(status='downloaded',bytes=len(response.content),sha256=hashlib.sha256(response.content).hexdigest())
 except Exception as error:row.update(status='unavailable',error=str(error))
 return row
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(get,URLS))
 manifest={'data_cutoff':'2026-09-19','sources':results,'author_appendix':{'status':'acquired_through_connected_Google_Drive','file':'external/FedPut_onlineappendix_rfs2.pdf','url':'https://drive.google.com/file/d/1YA1jNNnWax_E8-QHpZwse7wOYjadf0JN/view','sha256':hashlib.sha256((ROOT/'external/FedPut_onlineappendix_rfs2.pdf').read_bytes()).hexdigest()},'model_inference':'not run; no authenticated Gemini/Flash generation capability in the available environment; no model was silently substituted'}
 (ROOT/'results/source_acquisition_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('\n'.join(f"{r['file']}: {r['status']}" for r in results))
