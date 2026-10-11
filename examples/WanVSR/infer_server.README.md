# infer_server.py — FlashVSR OpenAI 兼容常驻超分服务

基于 `infer_flashvsr_v1.1_tiny_long_video.py` 改造的常驻 GPU 视频超分 HTTP 服务，
采用 **OpenAI 兼容风格** API（`/v1/*`、Bearer 鉴权、统一响应/错误信封）。

- 启动时加载一次模型并常驻显存/内存，重复任务不重新加载权重。
- 通过 `POST /v1/files` 上传视频（或用服务器本地路径），`POST /v1/video/upscales` 异步排队、串行推理。
- 支持任务状态查询、取消、下载超分结果。
- 输出保留原视频音轨（ffmpeg 编码 + remux），保留原 FPS/时长。
- 输入 N 帧 → 输出 N 帧（补帧向上补齐，不丢帧）。

## 启动

在 `examples/WanVSR/` 目录下执行（模型目录默认用当前目录的 `FlashVSR-v1.1` 软链）。

### 依赖

需要先安装 Web 服务依赖（不在上游 `requirements.txt` 中）：

```bash
pip install fastapi uvicorn python-multipart
```

### 运行

```bash
cd /cx8k/fs101/wzp/code/0929/FlashVSR/examples/WanVSR
conda activate flashvsr

# 可选：设置 API Key 启用 Bearer 鉴权
export FLASHVSR_API_KEY=sk-flashvsr-demo
CUDA_VISIBLE_DEVICES=7 python infer_server.py
```

首次启动会加载模型（约数十秒），看到 `runtime ready: load_seconds=...` 后即可接受请求。

> 脚本会自动把 `torch/lib` 加入 `LD_LIBRARY_PATH`（`block_sparse_attn` 动态链接
> `libc10.so` 需要），并以正确环境重新 `exec` 一次，因此可直接 `python infer_server.py`
> 启动，无需手动设置。

配置项均可通过命令行参数或同名环境变量设置，优先级为 **命令行 > 环境变量 > 默认值**：

| 命令行参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--host` | `FLASHVSR_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `FLASHVSR_PORT` | `18302` | 监听端口 |
| `--api-key` | `FLASHVSR_API_KEY` | 空 | **设置后启用 Bearer 鉴权**；为空则不鉴权 |
| `--model-dir` | `FLASHVSR_MODEL_DIR` | `./FlashVSR-v1.1` | 模型权重目录 |
| `--work-dir` | `FLASHVSR_WORK_DIR` | `./.flashvsr_server` | 输入/任务工作目录 |
| `--output-dir` | `FLASHVSR_OUTPUT_DIR` | `<work-dir>/outputs` | 超分结果输出目录（可指向独立磁盘/缓存路径） |
| `--encoder` | `FLASHVSR_ENCODER` | `auto` | `auto`/`nvenc`/`libx264` |
| `--max-queue` | `FLASHVSR_MAX_QUEUE` | `8` | 最大排队任务数 |
| `--max-target-pixels` | `FLASHVSR_MAX_TARGET_PIXELS` | `14500000` | 目标像素安全上限（防 OOM） |
| `--tiled-dit` | `FLASHVSR_TILED_DIT` | 关 | 默认启用 **tiled_dit** 空间分块降显存（`FLASHVSR_TILED_DIT=1`） |
| `--tile-size` | `FLASHVSR_TILE_SIZE` | `256` | tiled_dit 瓦片边长（**原始像素**） |
| `--tile-overlap` | `FLASHVSR_TILE_OVERLAP` | `24` | tiled_dit 瓦片重叠像素 |
| `--encode-chunk` | `FLASHVSR_ENCODE_CHUNK` | `4` | 编码时每次写入 ffmpeg 的帧数 |
| `--log-level` | `FLASHVSR_LOG_LEVEL` | `INFO` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR` |

```bash
# 等价写法：命令行覆盖环境变量
python infer_server.py --host 0.0.0.0 --port 18302 --api-key sk-flashvsr-demo \
  --model-dir ./FlashVSR-v1.1 --encoder auto --max-queue 8 --log-level INFO

python infer_server.py --help   # 查看全部参数
```

## 鉴权

- 设置 `FLASHVSR_API_KEY` 后，所有 `/v1/*` 请求都必须带：

  ```
  Authorization: Bearer <FLASHVSR_API_KEY>
  ```

- 未设置 `FLASHVSR_API_KEY` 时不鉴权（本地开发友好）。
- `/health` 永不鉴权；其响应的 `auth` 字段表示鉴权是否开启。
- 鉴权失败返回 `401`，错误体为 OpenAI 风格：

  ```json
  {"error": {"message": "Invalid API key", "type": "authentication_error", "param": null, "code": "invalid_api_key"}}
  ```

## API

Base URL：`http://127.0.0.1:18302`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（无需鉴权） |
| GET | `/v1/models` | 列出可用模型 |
| GET | `/v1/models/{model}` | 查询单个模型 |
| POST | `/v1/files` | 上传视频，返回 `file_id` |
| GET | `/v1/files` | 列出已上传文件 |
| DELETE | `/v1/files/{file_id}` | 删除已上传文件 |
| POST | `/v1/video/upscales` | 提交超分任务（`202`） |
| GET | `/v1/video/upscales` | 任务列表（最近 10 个） |
| GET | `/v1/video/upscales/{id}` | 任务状态 |
| POST | `/v1/video/upscales/{id}/cancel` | 取消任务 |
| GET | `/v1/video/upscales/{id}/content` | 下载超分 MP4 |

### 模型列表（`/v1/models`）

```json
{
  "object": "list",
  "data": [
    {"id": "FlashVSR-v1.1-Tiny-Long", "object": "model", "created": 1789546909, "owned_by": "flashvsr"}
  ]
}
```

`GET /v1/models/{model}` 查询单个模型；未知模型返回 `404 model_not_found`。

### 快速开始：上传 → 超分 → 下载

```bash
BASE=http://10.24.9.2:18302
AUTH="Authorization: Bearer sk-flashvsr-demo"   # 未设置 API_KEY 时删掉此变量

# 1. 上传视频，得到 file_id
UPLOAD=$(curl -fsS -X POST -H "$AUTH" \
  -F 'file=@/cx8k/fs101/wzp/code/0929/FlashVSR/results/bg_video_720p.mp4' \
  $BASE/v1/files)
FILE_ID=$(echo "$UPLOAD" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "FILE_ID=$FILE_ID"

# 2. 提交超分任务
JOB=$(curl -fsS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{\"file_id\":\"$FILE_ID\",\"scale\":2}" \
  $BASE/v1/video/upscales)
UPSCALE_ID=$(echo "$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "UPSCALE_ID=$UPSCALE_ID"

# 3. 轮询状态（json.tool 美化输出）
watch -n 2 "curl -fsS -H \"$AUTH\" $BASE/v1/video/upscales/$UPSCALE_ID | python -m json.tool"

# 4. 下载结果
curl -fL -H "$AUTH" "$BASE/v1/video/upscales/$UPSCALE_ID/content" -o output_upscaled.mp4

curl -fL -H "$AUTH" "$BASE/v1/video/upscales/$UPSCALE_ID/cancel"
```

### 快速开始：服务器本地路径

```bash
BASE=http://127.0.0.1:18302
AUTH="Authorization: Bearer sk-flashvsr-demo"

JOB=$(curl -fsS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"source":"/cx8k/fs101/wzp/code/0929/FlashVSR/bg_video_720p.mp4","scale":2}' \
  $BASE/v1/video/upscales)
UPSCALE_ID=$(echo "$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

# 轮询直到终止状态（queued → running → completed / failed / cancelled）
while :; do
  st=$(curl -fsS -H "$AUTH" $BASE/v1/video/upscales/$UPSCALE_ID \
        | python -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "status: $st"
  case "$st" in completed|failed|cancelled) break;; esac
  sleep 5
done

curl -fL -H "$AUTH" $BASE/v1/video/upscales/$UPSCALE_ID/content -o output_upscaled.mp4
```

### 文件对象（`/v1/files`）

上传响应与列表元素：

```json
{
  "id": "file_bf897d7f684a4d97b28c25e111ced756",
  "object": "file",
  "bytes": 6920602,
  "created_at": 1789546909,
  "filename": "bg_video_720p.mp4",
  "purpose": "upscale"
}
```

- `POST /v1/files` 为 multipart 上传，字段名 `file`。
- `GET /v1/files` 返回 `{"object": "list", "data": [...]}`，按上传时间倒序。
- `filename` 保留上传时的原始文件名（含扩展名），便于对应回源文件。
- `DELETE /v1/files/{file_id}` 返回 `{"id": "...", "object": "file", "deleted": true}`；若该文件正被未完成任务引用，返回 `409 file_in_use`。
- 上传文件在所有引用它的任务（含共享同一 `file_id` 的多个任务）都到达终态后才会被自动清理。

### 提交任务 body

```json
{
  "file_id": "file_...",   // 与 source 二选一
  "source": null,          // 服务器本地视频/图片目录路径（与 file_id 二选一）
  "model": null,           // 可选，OpenAI 兼容字段；填了则须等于模型名，否则 404 model_not_found
  "scale": 2.0,            // 1.0 ~ 4.0
  "seed": 0,
  "sparse_ratio": 2.0,     // 1.5 → 更快；2.0 → 更稳定（官方长视频默认）
  "local_range": 11,       // 9 → 更锐利；11 → 更稳定（官方长视频默认）
  "encoder": "auto",       // auto / nvenc / libx264
  "tiled_dit": false,      // 可选，空间分块跑 DiT 降显存（较慢）；缺省取服务端 FLASHVSR_TILED_DIT
  "tile_size": 256,        // 可选，瓦片边长（原始像素）；缺省取 FLASHVSR_TILE_SIZE
  "tile_overlap": 24       // 可选，瓦片重叠像素；缺省取 FLASHVSR_TILE_OVERLAP
}
```

未提供 `file_id` 或 `source` 之一、或同时提供，返回 `400`。

### tiled_dit（空间分块降显存）

高倍率（如 720p→4×）/长视频的显存峰值约为 **74 GB**，会在 80 GB 卡上 OOM
（`CUDA out of memory ... tried to allocate ...`）。根因是 **流式自注意力的 per-block KV 缓存
∝ 输出面积**（30 个 block 各缓存一份），不是碎片，所以 `expandable_segments` 无效。

`tiled_dit` 参考 [ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast)：
把输入帧在**空间上切成重叠瓦片**（`tile_size` 原始像素、`tile_overlap` 重叠）分别跑 DiT，
再用羽化权重（feather）融合回整幅。每块瓦片的 KV 缓存 ∝ 瓦片面积而非整幅，峰值大幅下降。

- 请求里 `"tiled_dit": true` 可单任务开启；或用 `FLASHVSR_TILED_DIT=1` / `--tiled-dit` 让服务默认开启。
- `tile_size`/`tile_overlap` 缺省用服务端配置。
- **代价**：逐瓦片串行推理，速度变慢；瓦片内容独立生成，画面与整幅推理**逐像素不完全一致**（但边界经羽化无接缝）。

实测（640×360→4×，180 帧，6 个 256 瓦片，H800）：

| | 峰值显存 | 耗时 | 输出 |
|---|---|---|---|
| 整幅（默认） | **32254 MB** | 48.8 s | 2560×1408 / 180 帧 |
| `tiled_dit`（tile=256, overlap=24） | **11878 MB** | 67.9 s | 2560×1408 / 180 帧 |

即 **峰值显存 ≈ 降到 1/2.7、耗时 ≈ 1.4×**。单瓦片（`tile_size` 覆盖整幅）时峰值与整幅一致、
PSNR 46.8 dB，验证分块管线正确；多瓦片与整幅 PSNR ≈ 24 dB（瓦片独立生成所致），
瓦片边界梯度能量 ≤ 全局（无接缝）。

### 任务对象（`/v1/video/upscales/{id}`）

`POST` 提交与 `GET` 查询返回同一信封：

```json
{
  "id": "upscale_6f83e57a4abf4eedb3bcf9f8ec034d2a_9f3c1a_2x",
  "object": "video.upscale",
  "model": "FlashVSR-v1.1-Tiny-Long",
  "file_id": "file_6f83e57a4abf4eedb3bcf9f8ec034d2a",
  "status": "queued",
  "phase": "waiting",
  "progress": 0.0,
  "created_at": 1789546909,
  "started_at": null,
  "finished_at": null,
  "queue_seconds": null,
  "elapsed_seconds": null,
  "total_seconds": 1.2,
  "queue_position": 1,
  "request": {
    "scale": 2.0, "seed": 0, "sparse_ratio": 2.0, "local_range": 11,
    "encoder": "auto", "source_name": "example4.mp4",
    "source_width": 1280, "source_height": 720,
    "source_frames": 822, "source_fps": 30.0, "source_duration_seconds": 27.4,
    "target_width": 2560, "target_height": 1408
  },
  "result": null,
  "error": null
}
```

- `status`：`queued | running | completed | failed | cancelled`
- `phase`：`waiting | preparing | upscaling | encoding | audio_remux | completed | failed | cancelled`
- `progress`：0~100 的当前进度百分比
- `queue_position`：仅 `queued` 时出现，排队位置（1 起）
- `file_id`：输入来源的 `file_id`；使用本地 `source` 提交时为 `null`
- `request` 含输入探测信息：`source_name`、`source_width/height`、**`source_frames`**、
  **`source_fps`**、**`source_duration_seconds`**（视频输入；图片目录提交时
  `source_fps`/`source_duration_seconds` 为 `null`），以及 `target_width/height`。
  轮询 `status` 时即可看到输入视频的时长/帧率/帧数。
- 各时间戳均为 **Unix 秒**（OpenAI 风格）
- 耗时字段（秒，均为数字或 `null`）：
  - `queue_seconds`：排队等待耗时（`started_at - created_at`；尚未开始为 `null`）。
  - `elapsed_seconds`：处理耗时；`running` 时按 `started_at` 实时递增，**成功**完成后等于
    `result.elapsed_seconds`（纯推理/编码时长），失败/取消则为 `finished_at - started_at`。
    队列中为 `null`。
  - `total_seconds`：从创建到完成（`finished_at - created_at`）的墙钟耗时；
    未完成时按当前时间实时递增。
- 完成时 `result`：

  ```json
  {"status": "completed", "scale": 2.0, "width": 2560, "height": 1408, "frames": 822,
   "fps": 30, "elapsed_seconds": 151.2, "peak_torch_allocated_mb": 12345,
   "encoder": "libx264", "resident": true, "engine_load_seconds": 52.2,
   "timings": {"prepare": 12.3, "inference": 120.1, "encode": 15.4, "audio_remux": 3.4}}
  ```

- 失败时 `error`：

  ```json
  {"message": "...", "type": "inference_error", "param": null, "code": null}
  ```

`phase` 流转：`preparing` 预处理 → `upscaling` 推理 → `encoding` 编码 → `audio_remux` 合并音轨。

### 任务 ID 与输入对齐

`upscale_id` 前缀直接取自输入指纹，便于从结果反查输入：

```
upscale_{输入指纹32hex}_{6位随机hex}_{scale}x
```

| 输入方式 | 指纹来源 | 示例 |
|---|---|---|
| 上传 `file_id` | 该 `file_id` 的 32 位 hex | `file_6f83e5...4d2a` + scale 2 → `upscale_6f83e5...4d2a_9f3c1a_2x` |
| 本地 `source` | 路径字符串的 sha256 前 32 位 | `/data/a.mp4` → `upscale_{sha256(path)[:32]}_9f3c1a_2x` |

- 前缀（32hex）与输入完全一致，一眼可对应；`file_id` 字段也一并暴露。
- 6 位随机后缀 + scale 标签保证同一输入多次提交（不同 scale/seed）互不冲突。

### 输出文件

- 存放目录：`FLASHVSR_OUTPUT_DIR`（默认 `./.flashvsr_server/outputs`）。
- 文件名：`{upscale_id}.mp4`，例如 `outputs/upscale_6f83e5...4d2a_9f3c1a_2x.mp4`。
- 服务在每个任务**入队**与**完成**时都会打印输出路径（完成时附带文件大小）：

  ```
  job upscale_6f83e5...4d2a_9f3c1a_2x queued: scale=2.0 seed=0 sparse_ratio=2.0 local_range=11 encoder=auto tiled_dit=False tile_size=256 tile_overlap=24 | input=... (1280x720, 822 frames, 30.0 fps, 27.4 s) | target=2560x1408 | out=/.../outputs/upscale_6f83e5...4d2a_9f3c1a_2x.mp4
  job upscale_6f83e5...4d2a_9f3c1a_2x completed: output=/.../outputs/upscale_6f83e5...4d2a_9f3c1a_2x.mp4 size=56584224B elapsed=219.0s encoder=libx264
  ```

### 任务列表（`/v1/video/upscales`）

返回 `{"object": "list", "data": [...]}`，**仅包含最近 10 个任务**，按 `created_at` 倒序。
列表项为任务对象的精简版（`id`/`object`/`model`/`file_id`/`status`/`phase`/`progress`/`scale`/`target_*`/时间戳/`queue_seconds`/`elapsed_seconds`/`total_seconds`/`error`）。

```bash
# 美化输出
curl -fsS -H "$AUTH" http://127.0.0.1:18302/v1/video/upscales | python -m json.tool

# jq 转紧凑表格（id | status | elapsed）
curl -fsS -H "$AUTH" http://127.0.0.1:18302/v1/video/upscales \
  | jq -r '.data[] | [.id[:24], .status, (.elapsed_seconds // "-"|tostring)] | @tsv' \
  | column -t -s$'\t'
# upscale_6f83e5...4d2a_9f3c1a_2x  completed  219.0
# upscale_e5f6a7...b8c9d0_1f2e3a_2x  failed     -
# upscale_9c0d1e...f2a3b4_c5d6e7_2x  queued     -
```

### 错误格式

所有错误（含参数校验）统一为 OpenAI 风格：

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "scale", "code": null}}
```

`type` 取值：`invalid_request_error`、`authentication_error`、`not_found_error`、
`rate_limit_exceeded`、`server_error`。常见状态码：

| 状态码 | 场景 |
|---|---|
| `400` | 参数非法（如 `file_id`/`source` 缺失或同时提供、视频无法解码） |
| `401` | 缺少/错误 API Key |
| `404` | 文件/任务不存在，或 `model` 不是已知模型（`model_not_found`） |
| `409` | 结果尚未就绪（`/content`），或删除正被使用的文件（`file_in_use`） |
| `422` | 目标分辨率超限、请求体校验失败 |
| `429` | 队列已满 |

## 说明

- 推理参数完全沿用官方脚本：`cfg_scale=1.0`、`num_inference_steps=1`、`kv_ratio=3.0`、
  `color_fix=True`、`is_full_block=False`、`if_buffer=True`、`topk_ratio=sparse_ratio*768*1280/(H*W)`。
- 单 GPU 串行执行：多个任务进入队列按顺序处理，避免并发采样导致 CUDA OOM。
- 目标像素超过 `FLASHVSR_MAX_TARGET_PIXELS` 会被拒绝（422），避免主机/显存 OOM。
- 取消立即生效：`POST /v1/video/upscales/{id}/cancel` 立刻返回 `cancelled` 状态；
  正在执行的**预处理 / 推理 / 编码**三个阶段都会在下一个帧/chunk 边界中断，
  不再跑完整迭代，并丢弃半成品输出（中断延迟约一个 chunk，通常数秒）。

## 日志

服务使用 Python `logging` 输出结构化日志，格式：

```
2025-09-16 12:00:00 | INFO    | flashvsr | uploaded file_id=file_6f83e5...4d2a filename=video.mp4 bytes=6908064
2025-09-16 12:00:01 | INFO    | flashvsr | job upscale_6f83e5...4d2a_9f3c1a_2x queued: scale=2.0 seed=0 sparse_ratio=2.0 local_range=11 encoder=auto tiled_dit=False tile_size=256 tile_overlap=24 | input=/.../inputs/file_6f83e5...4d2a.mp4 (1280x720, 822 frames, 30.0 fps, 27.4 s) | target=2560x1408 | out=/.../outputs/upscale_6f83e5...4d2a_9f3c1a_2x.mp4
2025-09-16 12:00:05 | INFO    | flashvsr | job upscale_6f83e5...4d2a_9f3c1a_2x started
2025-09-16 12:00:17 | INFO    | flashvsr | job upscale_6f83e5...4d2a_9f3c1a_2x upscaling 34.5% (elapsed 12s)
2025-09-16 12:01:30 | INFO    | flashvsr | job upscale_6f83e5...4d2a_9f3c1a_2x upscaling 76.2% (elapsed 85s)
2025-09-16 12:03:44 | INFO    | flashvsr | job upscale_6f83e5...4d2a_9f3c1a_2x completed: output=/.../outputs/upscale_6f83e5...4d2a_9f3c1a_2x.mp4 size=56584224B elapsed=219.0s encoder=libx264
```

日志级别通过 `FLASHVSR_LOG_LEVEL` 控制（默认 `INFO`）。关键日志事件：

| 级别 | 事件 |
|---|---|
| `INFO` | 服务启动/关闭、模型加载完成、文件上传/删除、任务排队（含输入/输出路径）/启动/**进度**/完成（含输出路径与大小）/取消、输入视频信息、音轨保留 |
| `WARNING` | 编码器回退（如 nvenc 不可用时自动切换 libx264）、remux 后音轨丢失 |
| `ERROR` | 任务失败（推理异常、未生成视频文件） |

取消任务时日志会区分两种情形：

- `cancelled during execution`：任务正在运行时被中断（三个阶段均支持，立即停止）。
- `cancelled (post-execution)`：任务在取消信号到达前已跑完，仅在收尾时被标记取消。
