import os, glob, argparse, yaml, sys
from collections import Counter
import pandas as pd

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_images(img_dir):
    exts = ('*.jpg','*.jpeg','*.png')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)

def inspect_split(split_dir, classes):
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    img_files = find_images(images_dir)
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"\n--- {os.path.basename(split_dir)} ---")
    print(f"Images: {len(img_files)}, Label files: {len(label_files)}")
    class_counts = Counter()
    malformed = []
    for lf in label_files:
        with open(lf, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    malformed.append((lf, ln))
                    continue
                try:
                    cls = int(parts[0])
                    if cls < 0 or cls >= len(classes):
                        malformed.append((lf, ln))
                    else:
                        class_counts[classes[cls]] += 1
                except:
                    malformed.append((lf, ln))
    print("Per-class counts:", dict(class_counts))
    if malformed:
        print(f"Found {len(malformed)} malformed label lines (showing first 5):")
        for m in malformed[:5]:
            print(m)
    return {"images": len(img_files), "labels": len(label_files), "class_counts": dict(class_counts), "malformed": malformed}

def main(args):
    data_yaml = os.path.join(args.data_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        print("data.yaml not found at", data_yaml)
        sys.exit(1)
    info = read_yaml(data_yaml)
    print("Loaded data.yaml:", data_yaml)
    print("Keys found:", list(info.keys()))
    classes = info.get('names') or info.get('nc') and list(range(info['nc']))
    if isinstance(classes, dict):
        # sometimes names are dict
        classes = [classes[k] for k in sorted(map(int, classes.keys()))]
    print("Classes:", classes)
    results = {}
    for split in ['train','valid','test']:
        split_path = os.path.join(args.data_dir, split)
        if os.path.exists(split_path):
            results[split] = inspect_split(split_path, classes)
        else:
            print(f"{split} folder not found (ok if you don't have it).")
    # Save summary
    df_rows = []
    for s, r in results.items():
        df_rows.append({'split': s, 'images': r['images'], 'label_files': r['labels'], 'malformed': len(r['malformed'])})
    df = pd.DataFrame(df_rows)
    out_csv = os.path.join(args.data_dir, 'dataset_summary.csv')
    df.to_csv(out_csv, index=False)
    print("\nSaved summary to", out_csv)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='dataset', help='root dataset dir with train/valid folders')
    p.add_argument('--show_sample', type=int, default=0, help='not used here; use visualize_samples.py')
    args = p.parse_args()
    main(args)
