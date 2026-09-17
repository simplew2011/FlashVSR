#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常驻 GPU 的 FlashVSR v1.1 Tiny-Long 视频超分服务（OpenAI 兼容 API）。

基于官方 infer_flashvsr_v1.1_tiny_long_video.py 改造：
  - 启动时加载一次模型并常驻显存/内存，重复任务不重新加载权重；
  - OpenAI 兼容风格 API：/v1/files 上传、/v1/video/upscales 提交/查询/取消/下载；
  - Bearer Token 鉴权（设置 FLASHVSR_API_KEY 后启用）；
  - 异步排队、串行推理；输出保留原视频音轨（ffmpeg 编码 + remux）。

推理参数完全沿用官方脚本（cfg_scale=1.0、num_inference_steps=1、kv_ratio=3.0、
color_fix=True、is_full_block=False、if_buffer=True、topk_ratio=sparse_ratio*768*1280/(H*W)）。
默认 sparse_ratio=2.0、local_range=11（官方长视频默认值）。

输入帧数 N → 输出帧数 N：补帧方式由官方脚本的"向下截断"改为"向上补齐"，
保证不丢帧。
"""
from __future__ import annotations

import gc
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import imageio
import numpy as np
import torch
from PIL import Image

# block_sparse_attn（由 diffsynth 间接导入）动态链接 torch 的 libc10.so，
# 需保证 LD_LIBRARY_PATH 在进程启动前就包含 torch/lib（glibc 启动时缓存该变量）。
# 若缺失则重新以正确环境 exec 一次，使脚本可直接 `python infer_server.py` 启动。
_TORCH_LIB = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
if _TORCH_LIB not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _TORCH_LIB + (
        ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)

from diffsynth import FlashVSRTinyLongPipeline, ModelManager
import diffsynth.pipelines.flashvsr_tiny_long as _ftl
from utils.TCDecoder import build_tcdecoder
from utils.utils import Causal_LQ4x_Proj

import asyncio
import uvicorn
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# 路径与配置
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "FlashVSR-v1.1"
PROMPT_CONTEXT = BASE_DIR / "prompt_tensor" / "posi_prompt.pth"

HOST = os.environ.get("FLASHVSR_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASHVSR_PORT", "18302"))
MODEL_DIR = Path(os.environ.get("FLASHVSR_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
WORK_DIR = Path(os.environ.get("FLASHVSR_WORK_DIR", str(BASE_DIR / ".flashvsr_server")))
ENCODER = os.environ.get("FLASHVSR_ENCODER", "auto")
MAX_QUEUE = int(os.environ.get("FLASHVSR_MAX_QUEUE", "8"))
MAX_TARGET_PIXELS = int(os.environ.get("FLASHVSR_MAX_TARGET_PIXELS", "4500000"))
MAX_TARGET_PIXELS = int(os.environ.get("FLASHVSR_MAX_TARGET_PIXELS", "14500000"))
ENCODE_CHUNK = int(os.environ.get("FLASHVSR_ENCODE_CHUNK", "4"))
API_KEY = os.environ.get("FLASHVSR_API_KEY", "")
MODEL_NAME = "FlashVSR-v1.1-Tiny-Long"
MODEL_CREATED = int(time.time())

INPUT_DIR = WORK_DIR / "inputs"
OUTPUT_DIR = Path(os.environ.get("FLASHVSR_OUTPUT_DIR", str(WORK_DIR / "outputs")))
JOB_DIR = WORK_DIR / "jobs"
FILE_META_DIR = WORK_DIR / "file_meta"

UPSCALE_ID_RE = re.compile(r"^upscale_[0-9a-f]{32}(_[0-9a-f]{6}_[0-9.]+x)?$")
FILE_ID_RE = re.compile(r"^file_[0-9a-f]{32}$")
TERMINAL = {"completed", "failed", "cancelled"}


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("FLASHVSR_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("flashvsr")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.propagate = False

for directory in (INPUT_DIR, OUTPUT_DIR, JOB_DIR, FILE_META_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _to_unix(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(str(ts)).timestamp())
    except (ValueError, TypeError):
        return None


def build_upscale_id(file_id: Optional[str], source: str, scale: float) -> str:
    """生成与输入对齐的 upscale_id：upscale_{输入指纹32hex}_{6随机hex}_{scale}x。

    上传文件用其 file_id 的 32 位 hex 作为指纹，本地路径用路径 sha256 前 32 位，
    使输出 id 前缀可直接反查输入；6 位随机后缀保证同一输入多任务不冲突。
    """
    if file_id is not None and FILE_ID_RE.fullmatch(file_id):
        prefix = file_id[len("file_"):]
    else:
        prefix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
    return f"upscale_{prefix}_{uuid.uuid4().hex[:6]}_{scale:g}x"


# --------------------------------------------------------------------------
# OpenAI 兼容错误 + 鉴权
# --------------------------------------------------------------------------
class OpenAIHTTPException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail={"message": message, "type": type, "param": param, "code": code},
        )


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not authorization:
        raise OpenAIHTTPException(401, "Missing Authorization header", "authentication_error", code="invalid_api_key")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer" or not hmac.compare_digest(parts[1], API_KEY):
        raise OpenAIHTTPException(401, "Invalid API key", "authentication_error", code="invalid_api_key")


# --------------------------------------------------------------------------
# 官方脚本的工具函数（原样保留）
# --------------------------------------------------------------------------
def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cpu"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple
    if tW == 0 or tH == 0:
        raise ValueError(f"Scaled size too small ({sW}x{sH}) for multiple={multiple}.")
    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    if tW > sW or tH > sH:
        raise ValueError(f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). Increase scale.")
    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def ceil_8n1(n: int) -> int:
    """最小的 >= n 且满足 8n+1（即 %8==1）的整数，用于向上补齐帧数，保证不丢帧。"""
    m = n
    while m % 8 != 1:
        m += 1
    return m


def prepare_input_tensor(path: str, scale: float = 4, dtype=torch.bfloat16,
                         cancel_check: Optional[Callable[[], bool]] = None):
    """解码并预处理输入，返回 (vid, tH, tW, padded_frames, fps, original_frames)。

    与官方脚本一致，逐帧用 PIL/imageio 解码、bicubic 放大后中心裁剪到 128 倍数，
    在 CPU 上累积 BF16 张量。区别：补帧改为向上补齐（ceil_8n1），不丢弃原始帧。
    逐帧检查 cancel_check，命中即抛 InferenceCancelled。
    """
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        total = len(paths0)
        name = os.path.basename(path)
        logger.info("[%s] Original Resolution: %dx%d | Original Frames: %d", name, w0, h0, total)

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        logger.info("[%s] Scaled (x%.2f): %dx%d -> Target (128-multiple): %dx%d", name, scale, sW, sH, tW, tH)

        padded = ceil_8n1(total + 4)
        paths = paths0 + [paths0[-1]] * (padded - total)
        logger.info("[%s] Padded Frames (8n+1): %d", name, padded)

        frames = []
        for p in paths:
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("prepare cancelled")
            with Image.open(p).convert("RGB") as img:
                frames.append(pil_to_tensor_neg1_1(upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH), dtype, "cpu"))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, padded, 30, total

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert("RGB")
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        name = os.path.basename(path)
        logger.info("[%s] Original Resolution: %dx%d | Original Frames: %d | FPS: %d", name, w0, h0, total, fps)

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        logger.info("[%s] Scaled (x%.2f): %dx%d -> Target (128-multiple): %dx%d", name, scale, sW, sH, tW, tH)

        padded = ceil_8n1(total + 4)
        idx = list(range(total)) + [total - 1] * (padded - total)
        logger.info("[%s] Padded Frames (8n+1): %d", name, padded)

        frames = []
        try:
            for i in idx:
                if cancel_check is not None and cancel_check():
                    raise InferenceCancelled("prepare cancelled")
                img = Image.fromarray(rdr.get_data(i)).convert("RGB")
                frames.append(pil_to_tensor_neg1_1(upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH), dtype, "cpu"))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        return vid, tH, tW, padded, fps, total

    raise ValueError(f"Unsupported input: {path}")


# --------------------------------------------------------------------------
# 常驻 pipeline 初始化
# --------------------------------------------------------------------------
def init_pipeline(model_dir: Path, context_path: Path):
    logger.info("GPU: %s CUDA=%s", torch.cuda.get_device_name(0), torch.version.cuda)
    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([str(model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device="cuda")

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    lq_proj_path = model_dir / "LQ_proj_in.ckpt"
    if lq_proj_path.exists():
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_proj_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")

    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    missing = pipe.TCDecoder.load_state_dict(torch.load(model_dir / "TCDecoder.ckpt", map_location="cpu"), strict=False)
    logger.info("TCDecoder missing keys: %s", missing)

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    context = torch.load(context_path, map_location="cpu")
    pipe.init_cross_kv(context_tensor=context)
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


# --------------------------------------------------------------------------
# ffmpeg 编码 + 音轨 remux（参考 flashvsr-service）
# --------------------------------------------------------------------------
def probe_video_info(path: Path) -> dict[str, int]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    try:
        from fractions import Fraction
        fps = max(1, int(round(float(Fraction(rate)))))
    except (ValueError, ZeroDivisionError):
        fps = 30
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


def scaled_dimensions(width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))
    target_width = (scaled_width // 128) * 128
    target_height = (scaled_height // 128) * 128
    if target_width <= 0 or target_height <= 0:
        raise ValueError("scaled resolution is too small")
    return scaled_width, scaled_height, target_width, target_height


def _encoder_command(save_path: Path, width: int, height: int, fps: int, encoder: str) -> list[str]:
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
    ]
    if encoder == "nvenc":
        command += [
            "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr",
            "-cq", "17", "-b:v", "0", "-spatial-aq", "1", "-temporal-aq", "1",
            "-pix_fmt", "yuv420p",
        ]
    else:
        # command += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p"]
        # command += ["-c:v", "libx265", "-preset", "fast", "-crf", "28", "-pix_fmt", "yuv420p", "-tag:v" , "hvc1"]
    command += ["-movflags", "+faststart", str(save_path)]
    return command


def _encode_tensor(frames: torch.Tensor, save_path: Path, fps: int, count: int, encoder: str,
                   cancel_check: Optional[Callable[[], bool]] = None) -> None:
    height, width = int(frames.shape[2]), int(frames.shape[3])
    process = subprocess.Popen(_encoder_command(save_path, width, height, fps, encoder), stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for start in range(0, count, ENCODE_CHUNK):
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("encode cancelled")
            end = min(start + ENCODE_CHUNK, count)
            chunk = frames[:, start:end].permute(1, 2, 3, 0).float()
            chunk = chunk.add_(1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
            if chunk.device.type != "cpu":
                chunk = chunk.to("cpu")
            process.stdin.write(chunk.contiguous().numpy().tobytes())
            del chunk
        process.stdin.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"{encoder} encoder exited with code {return_code}")


def save_tensor_video(frames: torch.Tensor, save_path: Path, fps: int, max_frames: int,
                      preferred_encoder: str = "auto",
                      cancel_check: Optional[Callable[[], bool]] = None) -> tuple[int, str]:
    frames = frames.detach()
    count = min(int(frames.shape[1]), int(max_frames))
    encoders = [preferred_encoder]
    if preferred_encoder == "auto":
        encoders = ["libx264", "nvenc"]
    elif preferred_encoder == "nvenc":
        encoders.append("libx264")
    last_error: Optional[Exception] = None
    for encoder in encoders:
        try:
            save_path.unlink(missing_ok=True)
            _encode_tensor(frames, save_path, fps, count, encoder, cancel_check=cancel_check)
            return count, encoder
        except InferenceCancelled:
            save_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            last_error = exc
            logger.warning("encoder fallback %s: %s", encoder, exc)
    raise RuntimeError(f"video encoding failed: {last_error}")


def _has_audio(path: Path) -> bool:
    """判断文件是否含音频流（用于校验 remux 后音轨未丢失）。"""
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return bool(completed.stdout.strip())


def remux_audio(video_only: Path, source: Path, output: Path) -> None:
    source_has_audio = _has_audio(source)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(source),
         "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output)],
        check=True,
    )
    if source_has_audio and not _has_audio(output):
        logger.warning("audio lost during remux: source=%s output=%s", source.name, output.name)
    elif source_has_audio:
        logger.info("audio preserved from source (%s)", source.name)
    else:
        logger.info("source has no audio track (%s)", source.name)


def release_host_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# --------------------------------------------------------------------------
# 任务取消：推理/预处理/编码三个阶段在 chunk/帧循环中检查取消信号，
# 命中即抛 InferenceCancelled，避免取消后仍跑完整迭代。
# --------------------------------------------------------------------------
class InferenceCancelled(Exception):
    """任务被取消时，用于中断正在进行的推理/预处理/编码。"""


# --------------------------------------------------------------------------
# 推理进度上报：FlashVSRTinyLongPipeline 的流式循环用 `tqdm(range(N))` 遍历
# chunk，这里把该模块的 tqdm 换成计数包装器，每处理一个 chunk 就回调一次进度，
# 并在取下一个 chunk 前检查取消信号。
# --------------------------------------------------------------------------
class _InferenceProgress:
    def __init__(self) -> None:
        self.total = 0
        self.current = 0
        self.cb: Optional[Callable[[float, str], None]] = None
        self.cancel_check: Optional[Callable[[], bool]] = None


_inference_progress = _InferenceProgress()
_real_tqdm = _ftl.tqdm


class _ProgressTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self._tqdm = _real_tqdm(iterable, *args, **kwargs)
        try:
            _inference_progress.total = len(iterable)
        except TypeError:
            _inference_progress.total = 0
        _inference_progress.current = 0

    def __iter__(self):
        for item in self._tqdm:
            if _inference_progress.cancel_check is not None and _inference_progress.cancel_check():
                self._tqdm.close()
                raise InferenceCancelled("inference cancelled")
            yield item
            _inference_progress.current += 1
            if _inference_progress.cb is not None and _inference_progress.total:
                _inference_progress.cb(
                    10.0 + 80.0 * (_inference_progress.current / _inference_progress.total),
                    "upscaling",
                )


# --------------------------------------------------------------------------
# 常驻运行时（线程内执行单个超分任务）
# --------------------------------------------------------------------------
class FlashVSRRuntime:
    def __init__(self, model_dir: Path, context_path: Path) -> None:
        loaded_at = time.monotonic()
        self.pipe = init_pipeline(model_dir, context_path)
        self.load_seconds = round(time.monotonic() - loaded_at, 3)
        self.jobs_completed = 0
        logger.info("runtime ready: load_seconds=%.3f", self.load_seconds)

    def process(
        self,
        source: Path,
        output: Path,
        scale: float,
        sparse_ratio: float = 2.0,
        local_range: int = 11,
        seed: int = 0,
        encoder: str = "auto",
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        source = source.resolve()
        output = output.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if not 1.0 <= scale <= 4.0:
            raise ValueError("scale must be between 1 and 4")
        if encoder not in {"auto", "nvenc", "libx264"}:
            raise ValueError("unsupported encoder")

        def report(p: float, phase: str) -> None:
            if progress_cb is not None:
                progress_cb(p, phase)

        output.parent.mkdir(parents=True, exist_ok=True)
        video_only = output.with_suffix(".video-only.mp4")
        total_started = time.monotonic()
        timings: dict[str, float] = {}
        lq_video = generated = None
        try:
            report(1.0, "preparing")
            stage = time.monotonic()
            lq_video, target_h, target_w, padded_frames, fps, original_frames = prepare_input_tensor(
                str(source), scale=scale, dtype=torch.bfloat16, cancel_check=cancel_check
            )
            timings["prepare"] = round(time.monotonic() - stage, 3)
            logger.info("job input=%s target=%dx%d frames=%d fps=%d", source, target_w, target_h, original_frames, fps)

            torch.cuda.reset_peak_memory_stats()
            stage = time.monotonic()
            self.pipe.load_models_to_device(["dit", "vae"])
            _original_tqdm = _ftl.tqdm
            _ftl.tqdm = _ProgressTqdm
            _inference_progress.cb = progress_cb
            _inference_progress.cancel_check = cancel_check
            try:
                generated = self.pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
                    LQ_video=lq_video, num_frames=padded_frames, height=target_h, width=target_w,
                    is_full_block=False, if_buffer=True,
                    topk_ratio=sparse_ratio * 768 * 1280 / (target_h * target_w),
                    kv_ratio=3.0, local_range=local_range, color_fix=True,
                )
            finally:
                _ftl.tqdm = _original_tqdm
                _inference_progress.cb = None
                _inference_progress.cancel_check = None
            torch.cuda.synchronize()
            timings["inference"] = round(time.monotonic() - stage, 3)
            peak_allocated = torch.cuda.max_memory_allocated()

            # 模型输出帧数为 num_frames - 4（即 F-4，见官方 "8n-3" 注释）。向上补齐后
            # 输出帧数 >= 原始帧数，切除多余的补帧，使输出帧数严格等于输入帧数。
            if generated.shape[1] > original_frames:
                generated = generated[:, :original_frames, :, :]

            del lq_video
            lq_video = None
            if hasattr(self.pipe.denoising_model(), "LQ_proj_in"):
                self.pipe.denoising_model().LQ_proj_in.clear_cache()
            self.pipe.TCDecoder.clean_mem()
            torch.cuda.empty_cache()
            release_host_memory()

            report(90.0, "encoding")
            stage = time.monotonic()
            output_frames, used_encoder = save_tensor_video(
                generated, video_only, fps=fps, max_frames=original_frames,
                preferred_encoder=encoder, cancel_check=cancel_check,
            )
            timings["encode"] = round(time.monotonic() - stage, 3)

            del generated
            generated = None
            release_host_memory()

            report(98.0, "audio_remux")
            stage = time.monotonic()
            remux_audio(video_only, source, output)
            timings["audio_remux"] = round(time.monotonic() - stage, 3)

            report(100.0, "completed")
            elapsed = round(time.monotonic() - total_started, 3)
            self.jobs_completed += 1
            return {
                "status": "completed",
                "scale": scale, "width": target_w, "height": target_h,
                "frames": output_frames, "fps": fps, "elapsed_seconds": elapsed,
                "peak_torch_allocated_mb": round(peak_allocated / 1024 / 1024),
                "encoder": used_encoder, "resident": True,
                "engine_load_seconds": self.load_seconds, "timings": timings,
            }
        finally:
            if lq_video is not None:
                del lq_video
            if generated is not None:
                del generated
            if hasattr(self.pipe.denoising_model(), "LQ_proj_in"):
                self.pipe.denoising_model().LQ_proj_in.clear_cache()
            self.pipe.TCDecoder.clean_mem()
            video_only.unlink(missing_ok=True)
            torch.cuda.empty_cache()
            release_host_memory()


# --------------------------------------------------------------------------
# 请求模型与 Worker
# --------------------------------------------------------------------------
class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: Optional[str] = Field(default=None, description="通过 /v1/files 上传返回的 file_id")
    source: Optional[str] = Field(default=None, description="服务器本地视频/图片目录路径")
    model: Optional[str] = Field(default=None, description="模型名，OpenAI 兼容字段（可选，仅校验）")
    scale: float = Field(default=2.0, ge=1.0, le=4.0)
    seed: int = Field(default=0)
    sparse_ratio: float = Field(default=2.0, ge=0.5, le=4.0)
    local_range: int = Field(default=11, ge=3, le=33)
    encoder: Literal["auto", "libx264", "nvenc"] = "auto"


class Worker:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE)
        self.runtime: Optional[FlashVSRRuntime] = None
        self.runner_task: Optional[asyncio.Task] = None
        self.load_jobs()

    def load_jobs(self) -> None:
        for path in JOB_DIR.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job_id = str(job.get("id", ""))
            if not UPSCALE_ID_RE.fullmatch(job_id):
                continue
            if job.get("status") in {"queued", "running"}:
                job["status"] = "failed"
                job["phase"] = "failed"
                job["finished_at"] = now_iso()
                job["error"] = {"message": "server restarted before the job completed", "type": "server_error", "param": None, "code": None}
                atomic_json(path, job)
            self.jobs[job_id] = job

    def persist(self, job: dict[str, Any]) -> None:
        atomic_json(JOB_DIR / f"{job['id']}.json", job)

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": job.get("id"),
            "object": job.get("object", "video.upscale"),
            "model": job.get("model", MODEL_NAME),
            "file_id": job.get("file_id"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "progress": job.get("progress", 0.0),
            "created_at": _to_unix(job.get("created_at")),
            "started_at": _to_unix(job.get("started_at")),
            "finished_at": _to_unix(job.get("finished_at")),
            "request": dict(job.get("request") or {}),
            "result": dict(job["result"]) if isinstance(job.get("result"), dict) else None,
            "error": job.get("error"),
        }
        if payload["status"] == "queued":
            queued = [jid for jid in list(self.queue._queue) if self.jobs.get(jid, {}).get("status") == "queued"]
            try:
                payload["queue_position"] = queued.index(job["id"]) + 1
            except ValueError:
                payload["queue_position"] = 1
        return payload

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        request = job.get("request") or {}
        error = job.get("error")
        return {
            "id": job.get("id"),
            "object": job.get("object", "video.upscale"),
            "model": job.get("model", MODEL_NAME),
            "file_id": job.get("file_id"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "progress": job.get("progress", 0.0),
            "scale": request.get("scale"),
            "target_width": request.get("target_width"),
            "target_height": request.get("target_height"),
            "created_at": _to_unix(job.get("created_at")),
            "started_at": _to_unix(job.get("started_at")),
            "finished_at": _to_unix(job.get("finished_at")),
            "elapsed_seconds": self._elapsed(job),
            "error": error.get("message") if isinstance(error, dict) else error,
        }

    def summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        ordered = sorted(
            self.jobs.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        )
        return [self.summary(job) for _, job in ordered[:limit]]

    @staticmethod
    def _elapsed(job: dict[str, Any]) -> Optional[float]:
        result = job.get("result")
        if isinstance(result, dict) and result.get("elapsed_seconds") is not None:
            return round(float(result["elapsed_seconds"]), 1)
        started = job.get("started_at")
        if not started:
            return None
        try:
            return round(time.time() - datetime.fromisoformat(str(started)).timestamp(), 1)
        except (ValueError, TypeError):
            return None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.runtime = await loop.run_in_executor(None, FlashVSRRuntime, MODEL_DIR, PROMPT_CONTEXT)
        self.runner_task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        if self.runner_task:
            self.runner_task.cancel()
            try:
                await self.runner_task
            except asyncio.CancelledError:
                pass

    async def submit(self, request: UpscaleRequest) -> dict[str, Any]:
        if request.model is not None and request.model != MODEL_NAME:
            raise OpenAIHTTPException(404, f"model '{request.model}' not found",
                                      type="not_found_error", param="model", code="model_not_found")
        if (request.source is None) == (request.file_id is None):
            raise OpenAIHTTPException(400, "provide exactly one of 'file_id' or 'source'", param="file_id")

        if request.file_id is not None:
            if not FILE_ID_RE.fullmatch(request.file_id):
                raise OpenAIHTTPException(400, "invalid file_id", param="file_id")
            matches = list(INPUT_DIR.glob(f"{request.file_id}.*"))
            if not matches:
                raise OpenAIHTTPException(404, "uploaded file not found", type="not_found_error", param="file_id", code="file_not_found")
            source = matches[0]
            source_provider = "upload"
        else:
            assert request.source is not None
            source = Path(request.source)
            if not source.is_file() and not source.is_dir():
                raise OpenAIHTTPException(404, "source path not found", type="not_found_error", param="source", code="source_not_found")
            source_provider = "path"

        if source.is_file():
            try:
                info = await asyncio.to_thread(probe_video_info, source)
            except (subprocess.CalledProcessError, KeyError, ValueError):
                raise OpenAIHTTPException(400, "cannot read video dimensions", param="file_id")
            source_width, source_height = info["width"], info["height"]
        else:
            paths0 = list_images_natural(str(source))
            if not paths0:
                raise OpenAIHTTPException(404, "no images in directory", type="not_found_error", param="source")
            with Image.open(paths0[0]) as img:
                source_width, source_height = img.size

        _, _, target_width, target_height = scaled_dimensions(source_width, source_height, request.scale)
        target_pixels = target_width * target_height
        if target_pixels > MAX_TARGET_PIXELS:
            raise OpenAIHTTPException(
                422,
                f"target {target_width}x{target_height} exceeds MAX_TARGET_PIXELS",
                param="scale", code="target_too_large",
            )

        active_count = sum(1 for item in self.jobs.values() if item.get("status") not in TERMINAL)
        if active_count >= MAX_QUEUE:
            raise OpenAIHTTPException(429, "queue is full", type="rate_limit_exceeded", code="queue_full")

        job_id = build_upscale_id(request.file_id, str(source), request.scale)
        while job_id in self.jobs:
            job_id = build_upscale_id(request.file_id, str(source), request.scale)
        output = OUTPUT_DIR / f"{job_id}.mp4"
        job = {
            "id": job_id,
            "object": "video.upscale",
            "model": MODEL_NAME,
            "file_id": request.file_id,
            "status": "queued",
            "phase": "waiting",
            "progress": 0.0,
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "source_provider": source_provider,
            "input_path": str(source),
            "output_path": str(output),
            "request": {
                "scale": request.scale,
                "seed": request.seed,
                "sparse_ratio": request.sparse_ratio,
                "local_range": request.local_range,
                "encoder": request.encoder,
                "source_width": source_width,
                "source_height": source_height,
                "target_width": target_width,
                "target_height": target_height,
            },
            "result": None,
            "error": None,
        }
        self.jobs[job_id] = job
        self.persist(job)
        await self.queue.put(job_id)
        logger.info("job %s queued: scale=%.1f target=%dx%d input=%s output=%s",
                    job_id, request.scale, target_width, target_height, source, output)
        return self.public(job)

    def _maybe_cleanup_input(self, job: dict[str, Any]) -> None:
        """上传文件在所有引用它的任务都到达终态后才删除，避免共享 file_id 的任务互相破坏。"""
        if job.get("source_provider") != "upload":
            return
        input_path = job.get("input_path")
        if not input_path:
            return
        for other in self.jobs.values():
            if other.get("id") == job.get("id"):
                continue
            if other.get("input_path") == input_path and other.get("status") not in TERMINAL:
                return
        try:
            Path(input_path).unlink(missing_ok=True)
        except OSError:
            pass
        _remove_file_meta(Path(input_path).stem)

    async def run_loop(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                job = self.jobs.get(job_id)
                if not job or job.get("status") == "cancelled":
                    continue
                await self.execute(job)
            finally:
                self.queue.task_done()
                finished = self.jobs.get(job_id)
                if finished and finished.get("status") in TERMINAL:
                    self._maybe_cleanup_input(finished)

    async def execute(self, job: dict[str, Any]) -> None:
        started = time.monotonic()
        job.update(status="running", phase="preparing", progress=1.0, started_at=now_iso(), error=None)
        self.persist(job)
        request = job["request"]
        logger.info("job %s started: scale=%.1f target=%dx%d",
                    job["id"], request.get("scale", 0), request.get("target_width"), request.get("target_height"))

        def on_progress(progress: float, phase: str) -> None:
            job["progress"] = round(float(progress), 1)
            job["phase"] = phase

        def cancelled() -> bool:
            return job.get("status") == "cancelled"

        try:
            result = await asyncio.to_thread(
                self.runtime.process,
                Path(job["input_path"]),
                Path(job["output_path"]),
                float(request["scale"]),
                float(request.get("sparse_ratio", 2.0)),
                int(request.get("local_range", 11)),
                int(request.get("seed", 0)),
                str(request.get("encoder", "auto")),
                on_progress,
                cancelled,
            )
        except InferenceCancelled:
            Path(job["output_path"]).unlink(missing_ok=True)
            logger.info("job %s cancelled during execution", job["id"])
            return
        except Exception as exc:
            if job.get("status") == "cancelled":
                Path(job["output_path"]).unlink(missing_ok=True)
                logger.info("job %s cancelled during execution", job["id"])
                return
            err_msg = str(exc) or type(exc).__name__
            job.update(status="failed", phase="failed", finished_at=now_iso(),
                       error={"message": err_msg, "type": "inference_error", "param": None, "code": None})
            self.persist(job)
            logger.error("job %s failed: %s", job["id"], err_msg)
            return

        if job.get("status") == "cancelled":
            Path(job["output_path"]).unlink(missing_ok=True)
            self.persist(job)
            logger.info("job %s cancelled (post-execution)", job["id"])
            return
        if not Path(job["output_path"]).is_file():
            job.update(status="failed", phase="failed", finished_at=now_iso(),
                       error={"message": "runtime did not produce a video", "type": "server_error", "param": None, "code": None})
            self.persist(job)
            logger.error("job %s failed: runtime did not produce a video", job["id"])
            return
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        job.update(status="completed", phase="completed", progress=100.0, finished_at=now_iso(), result=result, error=None)
        self.persist(job)
        output_file = Path(job["output_path"])
        try:
            output_size = output_file.stat().st_size
        except OSError:
            output_size = 0
        logger.info("job %s completed: output=%s size=%dB elapsed=%.1fs encoder=%s",
                    job["id"], output_file, output_size, result["elapsed_seconds"], result.get("encoder"))

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        if job.get("status") in TERMINAL:
            return self.public(job)
        previous_phase = job.get("phase")
        job.update(status="cancelled", phase="cancelled", finished_at=now_iso(), error=None)
        self.persist(job)
        logger.info("job %s cancelled (was %s)", job_id, previous_phase)
        if previous_phase == "waiting":
            self._maybe_cleanup_input(job)
        return self.public(job)


worker = Worker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("server starting on %s:%d", HOST, PORT)
    await worker.start()
    yield
    logger.info("server shutting down")
    await worker.stop()


app = FastAPI(title="FlashVSR Resident Server", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.exception_handler(OpenAIHTTPException)
async def openai_exception_handler(_, exc: OpenAIHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = [str(part) for part in first.get("loc", []) if part != "body"]
    return JSONResponse(
        status_code=422,
        content={"error": {
            "message": first.get("msg", "invalid request"),
            "type": "invalid_request_error",
            "param": ".".join(loc) or None,
            "code": None,
        }},
    )


def get_upscale_or_404(upscale_id: str) -> dict[str, Any]:
    if not UPSCALE_ID_RE.fullmatch(upscale_id) or upscale_id not in worker.jobs:
        raise OpenAIHTTPException(404, "upscale not found", type="not_found_error", code="not_found")
    return worker.jobs[upscale_id]


def _file_meta_path(file_id: str) -> Path:
    return FILE_META_DIR / f"{file_id}.json"


def _write_file_meta(file_id: str, filename: str, created_at: int) -> None:
    atomic_json(_file_meta_path(file_id), {"filename": filename, "created_at": created_at})


def _read_file_meta(file_id: str) -> dict[str, Any]:
    try:
        return json.loads(_file_meta_path(file_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remove_file_meta(file_id: str) -> None:
    _file_meta_path(file_id).unlink(missing_ok=True)


def _file_object(path: Path) -> dict[str, Any]:
    stat = path.stat()
    meta = _read_file_meta(path.stem)
    return {
        "id": path.stem,
        "object": "file",
        "bytes": stat.st_size,
        "created_at": int(meta.get("created_at") or stat.st_mtime),
        "filename": meta.get("filename") or path.name,
        "purpose": "upscale",
    }


@app.get("/health")
async def health():
    active = sum(1 for item in worker.jobs.values() if item.get("status") == "running")
    queued = sum(1 for item in worker.jobs.values() if item.get("status") == "queued")
    return {
        "status": "ok",
        "service": "flashvsr",
        "model": MODEL_NAME,
        "mode": "resident",
        "resident": worker.runtime is not None,
        "active": active,
        "queued": queued,
        "max_queue": MAX_QUEUE,
        "scales": [1.0, 4.0],
        "auth": bool(API_KEY),
        "jobs": worker.summaries(),
    }


@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": "flashvsr",
        }],
    }


@app.get("/v1/models/{model_id}", dependencies=[Depends(require_auth)])
async def get_model(model_id: str):
    if model_id != MODEL_NAME:
        raise OpenAIHTTPException(404, f"model '{model_id}' not found",
                                  type="not_found_error", param="model", code="model_not_found")
    return {"id": MODEL_NAME, "object": "model", "created": MODEL_CREATED, "owned_by": "flashvsr"}


@app.post("/v1/files", dependencies=[Depends(require_auth)])
async def create_file(file: UploadFile = File(...)):
    file_id = "file_" + uuid.uuid4().hex
    original_name = file.filename or "upload.mp4"
    ext = Path(original_name).suffix or ".mp4"
    dest = INPUT_DIR / f"{file_id}{ext}"
    created_at = int(time.time())
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            out.write(chunk)
    _write_file_meta(file_id, original_name, created_at)
    logger.info("uploaded file_id=%s filename=%s bytes=%d", file_id, original_name, total)
    return {
        "id": file_id,
        "object": "file",
        "bytes": total,
        "created_at": created_at,
        "filename": original_name,
        "purpose": "upscale",
    }


@app.get("/v1/files", dependencies=[Depends(require_auth)])
async def list_files():
    paths = [p for p in INPUT_DIR.glob("*") if p.is_file() and FILE_ID_RE.fullmatch(p.stem)]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {"object": "list", "data": [_file_object(p) for p in paths]}


@app.delete("/v1/files/{file_id}", dependencies=[Depends(require_auth)])
async def delete_file(file_id: str):
    if not FILE_ID_RE.fullmatch(file_id):
        raise OpenAIHTTPException(404, "file not found", type="not_found_error", code="file_not_found")
    matches = list(INPUT_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise OpenAIHTTPException(404, "file not found", type="not_found_error", code="file_not_found")
    in_use = any(
        job.get("source_provider") == "upload"
        and job.get("status") not in TERMINAL
        and Path(str(job.get("input_path", ""))).stem == file_id
        for job in worker.jobs.values()
    )
    if in_use:
        raise OpenAIHTTPException(409, "file is in use by a running upscale", code="file_in_use")
    for path in matches:
        path.unlink(missing_ok=True)
    _remove_file_meta(file_id)
    logger.info("deleted file_id=%s", file_id)
    return {"id": file_id, "object": "file", "deleted": True}


@app.post("/v1/video/upscales", status_code=202, dependencies=[Depends(require_auth)])
async def create_upscale(request: UpscaleRequest):
    return await worker.submit(request)


@app.get("/v1/video/upscales", dependencies=[Depends(require_auth)])
async def list_upscales():
    return {"object": "list", "data": worker.summaries()}


@app.get("/v1/video/upscales/{upscale_id}", dependencies=[Depends(require_auth)])
async def get_upscale(upscale_id: str):
    return worker.public(get_upscale_or_404(upscale_id))


@app.post("/v1/video/upscales/{upscale_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_upscale(upscale_id: str):
    get_upscale_or_404(upscale_id)
    return await worker.cancel(upscale_id)


@app.get("/v1/video/upscales/{upscale_id}/content", dependencies=[Depends(require_auth)])
async def download_upscale(upscale_id: str):
    job = get_upscale_or_404(upscale_id)
    if job.get("status") != "completed":
        raise OpenAIHTTPException(409, "video is not ready", code="not_ready")
    return FileResponse(job["output_path"], media_type="video/mp4", filename=f"flashvsr_{upscale_id}.mp4")


if __name__ == "__main__":
    uvicorn.run(app,
                host=HOST,
                port=PORT,
                workers=1,
                access_log=False,
                server_header=False,
                date_header=False,
                log_level="info")
