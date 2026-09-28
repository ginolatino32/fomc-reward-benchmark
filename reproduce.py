"""Rebuild the final comparisons offline from retained inputs.

python reproduce.py
python reproduce.py --recover-sources   # requires exact source PDFs
python reproduce.py --build-documents   # additionally requires pandoc
"""
from pathlib import Path
import argparse,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--recover-sources',action='store_true');p.add_argument('--build-documents',action='store_true');a=p.parse_args()
 env=os.environ.copy()
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
 commands=[]
 if a.recover_sources:
  for filename in ['source/w26894.pdf','external/FedPut_onlineappendix_rfs2.pdf']:
   if not (ROOT/filename).exists():raise FileNotFoundError(f'Obtain the exact source specified in README.md: {filename}')
  commands += [['recover_published_counts.py'],['recover_published_returns.py'],['recover_published_attention.py']]
 market_cache=(ROOT/'source/market/yahoo_gspc_daily.csv').exists() and (ROOT/'source/market/SP500.csv').exists()
 if not market_cache:print('Price caches not present (see scripts/fetch_market_data.py); using the released meeting-level targets in data/alternative_return_targets.csv.',flush=True)
 commands += [['prepare_features.py']]+([['alternative_targets.py']] if market_cache else [])+[['evaluate_extension.py'],['summarize_extension.py'],['evaluate_author_counts.py'],['evaluate_author_counts.py','--target','author_published_return_pct'],['evaluate_extension.py','--target','price_log_inclusive','--core-only'],['evaluate_extension.py','--target','price_log_excl_effective_policy','--core-only'],['source_diagnostics.py'],['final_tables.py'],['verify_final.py']]
 if a.build_documents:commands += [['build_documents.py'],['build_documents.py','--supplement']]
 (ROOT/'logs').mkdir(exist_ok=True)
 for i,cmd in enumerate(commands):
  print(f'[{i+1}/{len(commands)}] {" ".join(cmd)}',flush=True)
  with (ROOT/'logs'/f'reproduce_{i:02d}_{Path(cmd[0]).stem}.log').open('w') as log:
   subprocess.run([sys.executable,str(ROOT/'code'/cmd[0]),*cmd[1:]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Reproduction complete. See results/final_verification.json and manuscript/tables/ (created by final_tables.py).')
if __name__=='__main__':main()
