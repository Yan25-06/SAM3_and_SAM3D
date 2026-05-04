# Local / server scripts

## 1. Prepare env file

```bash
cp scripts/server/sam3_server.env.example scripts/server/sam3_server.env
```

Cho máy local Ubuntu như:

```bash
svo@svo-B760M-GAMING-PLUS-WIFI-DDR4:~/Downloads/sam3$
```

nên dùng:

```bash
cp scripts/server/sam3_local.env.example scripts/server/sam3_server.env
```

Edit:

- `PROJECT_ROOT`
- `VENV_DIR`
- `DATA_ROOT`
- `OUT_ROOT`
- `CHECKPOINT_PATH`

## Scope

Tài liệu này chỉ dùng cho:

- `ODinW13`
- `Table 2` zero-shot / 10-shot
- `Table 3` text / visual / text+visual

Không dùng `RF100-VL` trong flow này.

## 2. Create Python env

```bash
bash scripts/server/setup_sam3_env.sh
```

Nếu máy local chưa có `python3-venv`:

```bash
sudo apt update
sudo apt install -y python3-venv
```

## 3. Warm the SAM 3 checkpoint cache

```bash
bash scripts/server/download_sam3_checkpoint.sh
```

Sau khi tải xong, hãy copy hoặc đổi tên checkpoint về đúng chỗ mà script sẽ dùng:

```bash
mkdir -p checkpoints
cp /duong_dan_ban_da_tai/sam3.pt checkpoints/sam3.pt
```

Các script run bên dưới sẽ mặc định dùng:

```bash
+trainer.model.checkpoint_path=${PROJECT_ROOT}/checkpoints/sam3.pt
```

## Disk estimate

Cho flow ODinW13-only:

- zip ODinW13: khoảng `2.09 GB`
- dữ liệu giải nén: khoảng `4-6 GB`
- Python env + torch + deps: khoảng `10-15 GB`
- checkpoint/cache SAM 3: khoảng `4-8 GB`
- logs + prediction dumps: khoảng `5-10 GB`

Khuyên dùng:

- tối thiểu: `30 GB`
- hợp lý: `40 GB`
- thoải mái: `50 GB`

## 4. Download ODinW13

```bash
bash scripts/server/download_odinw13.sh
```

## 5. Run Table 3 on ODinW13

```bash
bash scripts/server/run_table3_odinw.sh text
bash scripts/server/run_table3_odinw.sh visual
bash scripts/server/run_table3_odinw.sh text_visual
```

Aggregate:

```bash
bash scripts/server/aggregate_odinw.sh table3_odinw_text_only
bash scripts/server/aggregate_odinw.sh table3_odinw_visual_only
bash scripts/server/aggregate_odinw.sh table3_odinw_text_and_visual
```

## 5b. Run Table 3 on LVIS

Script mới hỗ trợ 3 mode `text`, `visual`, `text_visual` và chạy `bbox` evaluation bằng official `lvis`.

Biến môi trường mặc định:

- `GT_FILE=${DATA_ROOT}/lvis/annotations/lvis_v1_val.json`
- `IMG_DIR=${DATA_ROOT}/coco`

Ví dụ:

```bash
bash scripts/server/run_table3_lvis.sh text
bash scripts/server/run_table3_lvis.sh visual
bash scripts/server/run_table3_lvis.sh text_visual
```

Chạy cả 3 mode:

```bash
bash scripts/server/run_table3_lvis.sh all
```

Lưu ý:

- Cần cài package `lvis` trong env đang chạy.
- `IMG_DIR` nên là root COCO image chứa `train2017/` và `val2017/`.

## 6. Run Table 2 on ODinW13

Zero-shot:

```bash
bash scripts/server/run_table2_odinw_zero_shot.sh
bash scripts/server/aggregate_odinw.sh table2_odinw_zero_shot
```

10-shot:

```bash
bash scripts/server/run_table2_odinw_10shot.sh 300
bash scripts/server/aggregate_odinw.sh table2_odinw_10shot_seed300
```

Optional extra seeds:

```bash
bash scripts/server/run_table2_odinw_10shot.sh 30
bash scripts/server/run_table2_odinw_10shot.sh 3
```

## 7. Run Table 1 on SA-Co/Gold

Default paths:

- `SACO_GOLD_ANN=${DATA_ROOT}/saco_gold/annotations`
- `SACO_GOLD_METACLIP_IMG=${DATA_ROOT}/saco_gold/metaclip_images`
- `SACO_GOLD_SA1B_IMG=${DATA_ROOT}/saco_gold/sa1b_images`

Run all 7 SA-Co/Gold subsets and evaluate both `segm` and `bbox`:

```bash
bash scripts/server/run_table1_saco_gold.sh
```

Resume only the missing SA-Co/Gold subsets after an interrupted run:

```bash
bash scripts/run_table1_sacogold_missing.sh
```
