from pathlib import Path
import time,subprocess
r=Path(__file__).resolve().parents[1]
needed=[r/'results/uploaded_as_received/author_method_minutes_meeting_counts.csv',r/'results/dictionary_only_correction/author_method_minutes_meeting_counts.csv',r/'results/audited_reconstruction/meeting_counts.csv']
start=time.monotonic()
while not all(p.exists() for p in needed):
    if time.monotonic()-start>900:raise TimeoutError('Count extraction not complete; inspect logs')
    time.sleep(2)
subprocess.run(['python',str(r/'code/evaluate_counts.py')],check=True)
