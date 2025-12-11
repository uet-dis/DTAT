## Preprocess (`cli.cic2018.preprocessing`)
### Merge data, clean missing/inf values and de-duplication
```
$ python -m cli.cic2018.preprocessing.merge_clean -d
2025-08-21 12:55:16 - __main__ - INFO - [+] Found 10 CSV files
2025-08-21 12:55:16 - __main__ - INFO - [+] ===========================================
2025-08-21 12:55:16 - __main__ - INFO - [+] PHASE 1-2 PREPROCESSING PIPELINE STARTED
2025-08-21 12:55:16 - __main__ - INFO - [+] ===========================================
2025-08-21 12:55:16 - __main__ - INFO - [+] Detecting schema across CSV files...
2025-08-21 12:55:16 - __main__ - INFO - [+] Majority schema detected from 9/10 files
2025-08-21 12:55:16 - __main__ - INFO - [+] Majority schema: 80 columns
2025-08-21 12:55:16 - __main__ - INFO - [+] Found 1 files with different schemas:
2025-08-21 12:55:16 - __main__ - INFO -     - Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv: +4 extras, -0 missing
2025-08-21 12:55:16 - __main__ - INFO -       Extra: ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']
2025-08-21 12:55:16 - __main__ - INFO - [+] Phase 1: Merge CSV files with early cleaning (stream -> per label files)...
2025-08-21 12:55:16 - __main__ - INFO - [+] Processing: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
2025-08-21 12:55:24 - __main__ - INFO - [+] Applying feature selection to reduce memory usage...
2025-08-21 12:55:24 - __main__ - INFO - [+] Dropping columns: ['Timestamp', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']
<SNIP>
2025-08-21 13:20:35 - __main__ - INFO - [+] ===========================================
2025-08-21 13:20:35 - __main__ - INFO - [+] PHASE 1-2 PREPROCESSING COMPLETED!
2025-08-21 13:20:35 - __main__ - INFO - [+] ===========================================
2025-08-21 13:20:35 - __main__ - INFO - [+] Input files processed: 10
2025-08-21 13:20:35 - __main__ - INFO - [+] Per-label files written: 15 @ /dis/DS/minhtq/CIC-2018//clean_merged
<SNIP>
```
### Split train/test = 7/3
```
$ python -m cli.cic2018.preprocessing.split_clean_merged --test-size 0.3 --random-state 42

2025-08-26 12:03:49 - __main__ - INFO - [SPLIT-CM] benign: total=10567392 -> train=7397174, test=3170218
2025-08-26 12:03:58 - __main__ - INFO - [SPLIT-CM] bot: total=144535 -> train=101174, test=43361
2025-08-26 12:03:58 - __main__ - INFO - [SPLIT-CM] brute_force_-web: total=555 -> train=388, test=167
2025-08-26 12:03:58 - __main__ - INFO - [SPLIT-CM] brute_force_-xss: total=228 -> train=159, test=69
2025-08-26 12:04:08 - __main__ - INFO - [SPLIT-CM] ddos_attack-hoic: total=198861 -> train=139202, test=59659
2025-08-26 12:04:08 - __main__ - INFO - [SPLIT-CM] ddos_attack-loic-udp: total=1730 -> train=1211, test=519
2025-08-26 12:04:40 - __main__ - INFO - [SPLIT-CM] ddos_attacks-loic-http: total=575364 -> train=402754, test=172610
2025-08-26 12:04:42 - __main__ - INFO - [SPLIT-CM] dos_attacks-goldeneye: total=41406 -> train=28984, test=12422
2025-08-26 12:04:47 - __main__ - INFO - [SPLIT-CM] dos_attacks-hulk: total=145199 -> train=101639, test=43560
2025-08-26 12:04:48 - __main__ - INFO - [SPLIT-CM] dos_attacks-slowhttptest: total=55 -> train=38, test=17
2025-08-26 12:04:48 - __main__ - INFO - [SPLIT-CM] dos_attacks-slowloris: total=9908 -> train=6935, test=2973
2025-08-26 12:04:48 - __main__ - INFO - [SPLIT-CM] ftp-bruteforce: total=53 -> train=37, test=16
2025-08-26 12:04:53 - __main__ - INFO - [SPLIT-CM] infilteration: total=139775 -> train=97842, test=41933
2025-08-26 12:04:53 - __main__ - INFO - [SPLIT-CM] sql_injection: total=84 -> train=58, test=26
2025-08-26 12:04:59 - __main__ - INFO - [SPLIT-CM] ssh-bruteforce: total=94048 -> train=65833, test=28215
```
### Setup encoders
```
$ python -m cli.cic2018.preprocessing.setup_encoders --train-subdir train
```
## Majority handling
### Encode
```
```
### Fit and transform PCA + UMAP
### Benign compression
1. Train subset compression:
+ Density-aware filter
+ 
+ 
```
(/home/soc/minhtq/AWGAN-implementation/.conda) [soc@hanoi AWGAN-implementation]$ python -m cli.cic2018.major.benign_comp.density_aware_filter   --kmeans-k 1500 --kmeans-batch 10000   --small-cluster-frac 0.001 --edge-percentile 92   --large-keep-rate 0.03 --min-keep-per-large 50 --log-level DEBUG
2025-08-29 18:02:46 - __main__ - INFO - [FILTER] Loading embeddings from /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding.parquet
2025-08-29 18:02:48 - __main__ - INFO - [FILTER] Filtering 7393647 benign embeddings
2025-08-29 18:02:49 - resampling.undersampling.density_aware - INFO - [FILTER] Using MiniBatchKMeans for micro-clustering
2025-08-29 18:02:49 - resampling.undersampling.density_aware - INFO - [FILTER] Fitting MiniBatchKMeans
2025-08-29 18:03:04 - resampling.undersampling.density_aware - INFO - [FILTER] Micro-clustered 7393647 embeddings
2025-08-29 18:03:05 - resampling.undersampling.density_aware - DEBUG - [FILTER] Cluster distribution (top10): {947: 29488, 1194: 28877, 674: 25413, 816: 21408, 40: 18504, 1072: 17972, 1466: 17409, 924: 17061, 104: 16602, 430: 16508}
[FILTER] dist->centroid: 100%|███████████████████████████████████████████████████████████████████████| 7393647/7393647 [00:25<00:00, 284666.97it/s]
2025-08-29 18:03:31 - resampling.undersampling.density_aware - INFO - [FILTER] Calculated distances to centroid
[FILTER] keep small clusters: 100%|███████████████████████████████████████████████████████████████████████████| 1500/1500 [00:13<00:00, 108.93it/s]
2025-08-29 18:03:45 - resampling.undersampling.density_aware - DEBUG - [FILTER] Small clusters threshold=7394 | points kept: 4176583
[FILTER] keep edge points: 100%|███████████████████████████████████████████████████████████████████████████████| 1500/1500 [04:36<00:00,  5.42it/s]
2025-08-29 18:08:22 - resampling.undersampling.density_aware - DEBUG - [FILTER] Edge points kept at 92.0th percentile | newly added: 257502
[FILTER] downsample large clusters: 100%|█████████████████████████████████████████████████████████████████████| 1500/1500 [00:01<00:00, 757.00it/s]
2025-08-29 18:08:24 - resampling.undersampling.density_aware - DEBUG - [FILTER] Downsampled large clusters | newly added: 96669
2025-08-29 18:08:24 - resampling.undersampling.density_aware - INFO - [FILTER] Calculated density-aware keep mask | Kept 4530754 / 7393647 (61.28%)
2025-08-29 18:08:33 - __main__ - INFO - [FILTER] Saved filtered Benign embeddings -> /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered.parquet
(/home/soc/minhtq/AWGAN-implementation/.conda) [soc@hanoi AWGAN-implementation]$ python -m cli.cic2018.major.benign_comp.boundary_with_infil --budget-total 28000 --pin-frac 0.3 --float32 --log-level DEBUG                                                                                          2025-08-29 18:08:52 - __main__ - INFO - [BOUNDARY] Using density-aware filtered Benign: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-08-29 18:08:52 - __main__ - INFO - [BOUNDARY] Loading Benign embeddings: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-08-29 18:08:54 - __main__ - INFO - [BOUNDARY] Loading Infiltration embeddings: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_infilteration_embedding.parquet
2025-08-29 18:08:54 - __main__ - INFO - [BOUNDARY] Shapes | Benign=4530754 rows, Infiltration=97523 rows
2025-08-29 18:08:54 - __main__ - DEBUG - [BOUNDARY] Benign clusters: total=1500; top5={1368: 7389, 1345: 7362, 467: 7360, 478: 7343, 940: 7343}
2025-08-29 18:08:55 - __main__ - INFO - [BOUNDARY] Fitting NN on Infiltration: 97523
[BOUNDARY] knn (benign→infil): 100%|███████████████████████████████████████████████████████████████████████████████| 23/23 [00:54<00:00,  2.39s/it]
2025-08-29 18:09:50 - __main__ - INFO - [BOUNDARY] d(ben→inf): min=0.0000, mean=0.7024, p50=0.0479, p90=4.0149, p95=4.9590, max=11.2406
2025-08-29 18:09:50 - __main__ - INFO - [BOUNDARY] boundary_score: min=0.0817, mean=0.8171, p50=0.9543, p90=0.9962, p95=0.9984, max=1.0000
2025-08-29 18:09:55 - __main__ - INFO - [BOUNDARY] PIN boundary count: 8400 / 4530754 | cutoff boundary_score=1.000000
2025-08-29 18:09:55 - __main__ - DEBUG - [BOUNDARY] PIN distribution (top10 clusters): {695: 356, 549: 204, 260: 182, 178: 179, 769: 178, 1376: 156, 1027: 123, 231: 108, 943: 98, 1247: 94}
2025-08-29 18:10:03 - __main__ - INFO - [BOUNDARY] Saved annotated Benign -> /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered_with_boundary.parquet
2025-08-29 18:10:03 - __main__ - INFO - [BOUNDARY] Saved PIN subset -> /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered_pin.parquet
(/home/soc/minhtq/AWGAN-implementation/.conda) [soc@hanoi AWGAN-implementation]$ python -m cli.cic2018.major.benign_comp.coreset --float32 --budget-total 28000 --qc-enable --log-level DEBUG
2025-08-29 18:11:34 - __main__ - INFO - [CORESET] Using annotated filtered embeddings: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered_with_boundary.parquet
2025-08-29 18:11:37 - __main__ - INFO - [CORESET] PIN available: 8400
2025-08-29 18:11:37 - __main__ - INFO - [CORESET] Selecting 19600 coresets from non-PIN: 4522354 candidates
[CORESET] choose reps: 100%|████████████████████████████████████████████████████████████████████████████████| 19600/19600 [00:34<00:00, 561.48it/s]
2025-08-29 18:14:37 - __main__ - INFO - [CORESET] Final coreset size: 28000
[CORESET] dist core->PIN: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:09<00:00,  9.89s/it]
2025-08-29 18:14:48 - __main__ - INFO - [QC] Covering radius (nearest selected): p50=0.0260, p90=0.0940, p95=0.1296
2025-08-29 18:14:48 - __main__ - INFO - [QC] Cluster coverage: total=1500, kept=1500, small_clusters_present=150/150
2025-08-29 18:14:48 - __main__ - INFO - [QC] boundary_score in coreset: mean=0.8698, p50=0.9813, p90=1.0000, p95=1.0000
2025-08-29 18:14:48 - __main__ - INFO - [QC] PIN present in coreset: 8400
2025-08-29 18:14:48 - __main__ - INFO - [CORESET] Saved coreset -> /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered_with_boundary_compressed_coreset.parquet
```