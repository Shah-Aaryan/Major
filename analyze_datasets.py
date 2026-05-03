import os
from datetime import datetime

raw_dir = r'c:\Users\Naman\Desktop\SPIT\projects\Major\data\raw'
datasets = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and f.startswith('OHLCV')]

info = []
for ds in sorted(datasets):
    path = os.path.join(raw_dir, ds)
    size_mb = os.path.getsize(path) / (1024*1024)
    # Parse: OHLCV_Binance_TOKEN-USDT_D<start>-D<end>_1min.csv
    name = ds.replace('OHLCV_Binance_', '').replace('_1min.csv', '')
    # Split on _D
    idx = name.find('_D')
    token = name[:idx]
    date_part = name[idx+2:]  # remove leading _D
    parts = date_part.split('-D')
    start_raw = parts[0].split('T')[0] if parts else 'N/A'
    end_raw = parts[1].split('T')[0] if len(parts) > 1 else 'N/A'
    try:
        d1 = datetime.strptime(start_raw, '%Y%m%d')
        d2 = datetime.strptime(end_raw, '%Y%m%d')
        years = round((d2 - d1).days / 365, 1)
        start_fmt = d1.strftime('%Y-%m-%d')
        end_fmt = d2.strftime('%Y-%m-%d')
    except Exception as e:
        years = 0
        start_fmt = start_raw
        end_fmt = end_raw
    info.append({
        'token': token,
        'start': start_fmt,
        'end': end_fmt,
        'years': years,
        'size_mb': round(size_mb, 1),
        'filename': ds
    })

print(f"{'Token':<15} | {'Start':>10} | {'End':>10} | {'Years':>5} | {'Size MB':>8}")
print("-" * 70)
for i in sorted(info, key=lambda x: x['years'], reverse=True):
    print(f"{i['token']:<15} | {i['start']:>10} | {i['end']:>10} | {i['years']:>5} | {i['size_mb']:>8}")
