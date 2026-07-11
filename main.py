"""AstrBot 智能分段插件。"""
from __future__ import annotations
import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult
from astrbot.api.event import filter
from astrbot.api.message_components import Plain
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star

# 导入多段拼接自愈与本地切分兜底依赖
from .segmentation import (
    build_segmentation_prompt,
    calculate_send_delay,
    hash_normalized_text,
    is_action_only_text,
    parse_segments_from_model_output,
    strip_thinking_content,
    normalize_response_text_for_key,
    merge_segments_balancing_brackets,
    split_segments_at_bracket_boundaries,
    # 导入非自然语言检测与本地逻辑切分工具
    is_non_natural_language,
    local_fallback_split,
)

_PREPARED_SEGMENT_TTL_SECONDS = 60.0
_PENDING_FOLLOW_UP_TTL_SECONDS = 60.0
_PENDING_EXTRA_KEY = "smart_segmentation_pending_id"


@dataclass(slots=True)
class SegmentationSettings:
    enabled: bool = True
    provider_id: str = ""
    style: str = "natural"
    min_length: int = 15
    max_segments: int = 8
    temperature: float = 0.3
    max_tokens: int = 600
    timeout_seconds: float = 12.0
    delay_base: float = 0.35
    delay_per_char: float = 0.015
    delay_max: float = 1.2


@dataclass(slots=True)
class PreparedSegments:
    segments: list[str]
    raw_norm_text: str  # 缓存此分段对应的原始文本规范化表示，用作自愈拼合时的唯一基准
    expires_at: float


@dataclass(slots=True)
class PendingFollowUp:
    session: str
    segments: list[str]
    delay_base: float
    delay_per_char: float
    delay_max: float
    expires_at: float


class SmartSegmentationPlugin(Star):
    """使用 LLM 对 AstrBot 主回复进行自然分段。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        """初始化智能分段插件并设置初始缓存状态。"""
        super().__init__(context)
        self.config = config if config is not None else {}
        self._prepared_segments: dict[tuple[str, str], PreparedSegments] = {}
        self._pending_follow_ups: dict[str, PendingFollowUp] = {}
        self._active_follow_up_tasks: set[asyncio.Task[Any]] = set()
        self._send_guards: dict[str, int] = {}

    @filter.on_llm_response()
    async def on_llm_response(
        self,
        event: AstrMessageEvent,
        response: LLMResponse,
    ) -> None:
        """在主 LLM 返回后预先计算智能分段。"""
        settings = self._get_settings()
        if settings is None:
            return

        text = self._extract_response_plain_text(response)
        if not self._should_segment_text(text, settings):
            return

        # 在 LLM 处理前拦截非自然语言（如 JSON、代码块），防止小参数模型产生 Few-Shot 样例幻觉
        if is_non_natural_language(text):
            logger.info("检测到非自然语言（JSON/代码等），跳过 LLM 调用，直接启用本地高精切分兜底")
            segments = local_fallback_split(text, settings.max_segments)
        else:
            provider_id = await self._resolve_provider_id(event, settings)
            if not provider_id:
                logger.warning("智能分段未找到可用 provider_id，跳过本次分段")
                return

            try:
                segments = await asyncio.wait_for(
                    self._segment_text(
                        text,
                        provider_id=provider_id,
                        settings=settings,
                    ),
                    timeout=settings.timeout_seconds,
                )
            except TimeoutError:
                logger.warning(
                    "智能分段 LLM 调用超时（> %.2fs），已跳过本次回复",
                    settings.timeout_seconds,
                )
                return
            except Exception as exc:
                logger.error("智能分段 LLM 调用失败: %s", exc, exc_info=True)
                return

        if not segments or len(segments) <= 1:
            return

        self._store_prepared_segments(event.unified_msg_origin, text, segments)
        logger.info("智能分段预处理完成，共 %s 段", len(segments))

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """发送前把完整回复替换为首段，并登记剩余段。"""
        settings = self._get_settings()
        if settings is None:
            return

        if self._is_session_guarded(event.unified_msg_origin):
            return

        result = event.get_result()
        if result is None or not self._is_model_text_result(result):
            return

        outbound_text = self._extract_plain_text_chain(result)
        if not outbound_text:
            return

        # 传入最大分段数与最小长度参数，以支持自愈拼接决策及本地高精兜底切分
        segments = self._pop_prepared_segments(
            event.unified_msg_origin, 
            outbound_text,
            min_length=settings.min_length,
            max_segments=settings.max_segments
        )
        if not segments or len(segments) <= 1:
            return

        first_segment = segments[0]
        follow_up_segments = segments[1:]
        if not follow_up_segments:
            return

        result.chain = [Plain(first_segment)]
        pending_id = self._register_pending_follow_up(
            session=event.unified_msg_origin,
            segments=follow_up_segments,
            settings=settings,
        )
        event.set_extra(_PENDING_EXTRA_KEY, pending_id)
        logger.info("智能分段首段已替换，登记 %s 条补发消息", len(follow_up_segments))

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        """首段发送后后台补发剩余分段。"""
        pending_id = str(event.get_extra(_PENDING_EXTRA_KEY, "") or "").strip()
        if not pending_id:
            return

        pending = self._pop_pending_follow_up(pending_id)
        if pending is None or not pending.segments:
            return

        task = asyncio.create_task(self._run_follow_up_segments(pending))
        self._track_follow_up_task(task)

    async def terminate(self) -> None:
        """插件卸载时取消尚未完成的补发任务并清空缓存。"""
        for task in list(self._active_follow_up_tasks):
            if not task.done():
                task.cancel()
        await self._drain_tasks()
        self._active_follow_up_tasks.clear()
        self._prepared_segments.clear()
        self._pending_follow_ups.clear()
        self._send_guards.clear()

    def _get_config_value(self, key: str, default: Any) -> Any:
        """从插件配置中安全获取指定键的值，提供降级默认值。"""
        try:
            if hasattr(self.config, "get"):
                return self.config.get(key, default)
        except Exception as exc:
            logger.debug("读取智能分段配置 %s 失败: %s", key, exc)
        return default

    def _get_settings(self) -> SegmentationSettings | None:
        """根据插件当前配置，解析并构建类型安全的 `SegmentationSettings` 实例。"""
        enabled = self._as_bool(self._get_config_value("enabled", True), True)
        if not enabled:
            return None

        style = str(self._get_config_value("style", "natural") or "natural").strip()
        if style not in {"natural", "conservative", "active"}:
            style = "natural"

        return SegmentationSettings(
            enabled=enabled,
            provider_id=str(self._get_config_value("provider_id", "") or "").strip(),
            style=style,
            min_length=max(
                0,
                self._as_int(self._get_config_value("min_length", 15), 15),
            ),
            max_segments=max(
                1,
                self._as_int(self._get_config_value("max_segments", 8), 8),
            ),
            temperature=self._as_float(
                self._get_config_value("temperature", 0.3),
                0.3,
            ),
            max_tokens=max(
                1,
                self._as_int(self._get_config_value("max_tokens", 600), 600),
            ),
            timeout_seconds=max(
                0.1,
                self._as_float(
                    self._get_config_value("timeout_seconds", 12.0),
                    12.0,
                ),
            ),
            delay_base=max(
                0.0,
                self._as_float(self._get_config_value("delay_base", 0.35), 0.35),
            ),
            delay_per_char=max(
                0.0,
                self._as_float(
                    self._get_config_value("delay_per_char", 0.015),
                    0.015,
                ),
            ),
            delay_max=max(
                0.0,
                self._as_float(self._get_config_value("delay_max", 1.2), 1.2),
            ),
        )

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        """将输入值安全转换为布尔值，兼容字符串真伪表达。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        """将输入值转换为整型，若失败则安全降级到默认值。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        """将输入值转换为浮点数，若失败则安全降级到默认值。"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_response_plain_text(response: LLMResponse) -> str:
        """从大语言模型响应结构中提取纯文本，自动过滤思考标签。"""
        role = str(getattr(response, "role", "") or "").strip().lower()
        if role and role not in {"assistant", "ai"}:
            return ""

        result_chain = getattr(response, "result_chain", None)
        if result_chain is not None and not SmartSegmentationPlugin._is_plain_chain(
            result_chain,
        ):
            return ""

        text = str(getattr(response, "completion_text", "") or "")
        return strip_thinking_content(text)

    @staticmethod
    def _is_plain_chain(message_chain: MessageChain) -> bool:
        """判断消息链是否为纯文本组成的轻量级消息链。"""
        chain = getattr(message_chain, "chain", None)
        return isinstance(chain, list) and bool(chain) and all(
            isinstance(component, Plain) for component in chain
        )

    @classmethod
    def _extract_plain_text_chain(cls, message_chain: MessageChain) -> str:
        """从消息事件结果中提取并整合干净的纯文本内容。"""
        if not cls._is_plain_chain(message_chain):
            return ""
        texts = [component.text for component in message_chain.chain]
        return strip_thinking_content(" ".join(texts))

    @classmethod
    def _is_model_text_result(cls, result: MessageEventResult) -> bool:
        """判断当前消息装饰结果是否为大模型生成的纯文本响应。"""
        is_model_result = getattr(result, "is_model_result", None)
        if callable(is_model_result):
            try:
                if not is_model_result():
                    return False
            except Exception:
                return False
        return cls._is_plain_chain(result)

    @staticmethod
    def _should_segment_text(text: str, settings: SegmentationSettings) -> bool:
        """根据文本长度及特征，判定当前文本是否需要执行分段。"""
        if not text:
            return False
        if len(text) < settings.min_length:
            return False
        return not is_action_only_text(text)

    async def _resolve_provider_id(
        self,
        event: AstrMessageEvent,
        settings: SegmentationSettings,
    ) -> str:
        """获取当前消息事件对应会话的底层 LLM 提供商 ID。"""
        if settings.provider_id:
            return settings.provider_id

        get_current = getattr(self.context, "get_current_chat_provider_id", None)
        if callable(get_current):
            try:
                provider_id = await get_current(event.unified_msg_origin)
                if provider_id:
                    return str(provider_id).strip()
            except Exception as exc:
                logger.debug("获取当前会话 provider_id 失败: %s", exc)

        get_using = getattr(self.context, "get_using_provider", None)
        if callable(get_using):
            try:
                provider = get_using(event.unified_msg_origin)
                meta = provider.meta() if provider and hasattr(provider, "meta") else None
                provider_id = getattr(meta, "id", "") if meta else ""
                return str(provider_id or "").strip()
            except Exception as exc:
                logger.debug("回退获取 provider_id 失败: %s", exc)

        return ""

    async def _segment_text(
        self,
        text: str,
        *,
        provider_id: str,
        settings: SegmentationSettings,
    ) -> list[str]:
        """调用大模型并传入针对性提示词，执行智能分段流程。"""
        prompt = build_segmentation_prompt(text, settings.style, settings.max_segments)
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        raw_text = str(getattr(response, "completion_text", "") or "").strip()
        if not raw_text:
            return [text]

        return parse_segments_from_model_output(
            raw_text,
            fallback_text=text,
            max_segments=settings.max_segments,
        )

    def _store_prepared_segments(
        self,
        session: str,
        response_text: str,
        segments: list[str],
    ) -> None:
        """将预处理的分段结果及原始文本的特征摘要写入高速缓存。"""
        self._prune_expired_prepared_segments()
        text_hash = hash_normalized_text(response_text)
        normalized_session = str(session or "").strip()
        if not normalized_session or not text_hash:
            return

        # 缓存分段结果的同时，缓存其原始文本的规范化特征，作为未来恢复拼接拼合的基准
        self._prepared_segments[(normalized_session, text_hash)] = PreparedSegments(
            segments=list(segments),
            raw_norm_text=normalize_response_text_for_key(response_text),
            expires_at=time.monotonic() + _PREPARED_SEGMENT_TTL_SECONDS,
        )

    def _pop_prepared_segments(
        self, 
        session: str, 
        outbound_text: str,
        *,
        min_length: int = 15,
        max_segments: int = 0,
    ) -> list[str] | None:
        """从预存库中获取分段结果（精确匹配 -> 顺序贪婪组合自愈 -> 本地高精分段兜底）。"""
        self._prune_expired_prepared_segments()
        normalized_session = str(session or "").strip()
        norm_outbound = normalize_response_text_for_key(outbound_text)
        if not normalized_session or not norm_outbound:
            return None

        # 1. 精确哈希匹配
        text_hash = hash_normalized_text(outbound_text)
        entry = self._prepared_segments.pop((normalized_session, text_hash), None)
        if entry is not None:
            return list(entry.segments)

        # 2. 顺序贪婪自愈（应对多轮 Tool Loop 被框架拼接起来合并发送的情况）
        # 收集该会话中当前缓存的全部可用缓存项
        session_entries = {
            key: val
            for key, val in self._prepared_segments.items()
            if key[0] == normalized_session
        }
        if session_entries:
            remaining = norm_outbound.strip()
            assembled_segs = []
            matched_keys = []
            while remaining:
                best_match_len = 0
                best_match_segs = None
                best_match_key = None
                for key, val in session_entries.items():
                    norm_piece = val.raw_norm_text
                    if norm_piece and remaining.startswith(norm_piece):
                        piece_length = len(norm_piece)
                        if piece_length > best_match_len:
                            # 保证是在空格边界或结束边界上匹配，防止单词中继切分错乱
                            if piece_length == len(remaining) or remaining[piece_length] == ' ':
                                best_match_len = piece_length
                                best_match_segs = val.segments
                                best_match_key = key
                if best_match_segs and best_match_key:
                    assembled_segs.extend(best_match_segs)
                    remaining = remaining[best_match_len:].strip()
                    matched_keys.append(best_match_key)
                else:
                    break

            if not remaining and assembled_segs:
                # 完美复原合并消息！清空这部分已消费的子缓存，避免残留过期
                for m_key in matched_keys:
                    self._prepared_segments.pop(m_key, None)
                return assembled_segs

        # 3. 终极自愈兜底：本地纯逻辑高精切分
        if len(outbound_text) >= min_length and not is_action_only_text(outbound_text):
            # 直接复用 segmentation.py 中封装的高精度切分函数
            local_segs = local_fallback_split(outbound_text, max_segments)
            if local_segs and len(local_segs) > 1:
                logger.info("智能分段精准哈希/拼合未中，已自动降级启动本地高精切分兜底")
                return local_segs

        return None

    def _prune_expired_prepared_segments(self) -> None:
        """清理缓存池中生命周期已结束的预存分段条目，防止内存溢出。"""
        if not self._prepared_segments:
            return
        now = time.monotonic()
        expired_keys = [
            key
            for key, entry in self._prepared_segments.items()
            if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._prepared_segments.pop(key, None)

    def _register_pending_follow_up(
        self,
        *,
        session: str,
        segments: list[str],
        settings: SegmentationSettings,
    ) -> str:
        """登记当前会话中等待后台补发的所有后续分段消息。"""
        self._prune_expired_pending_follow_ups()
        pending_id = uuid4().hex
        self._pending_follow_ups[pending_id] = PendingFollowUp(
            session=session,
            segments=list(segments),
            delay_base=settings.delay_base,
            delay_per_char=settings.delay_per_char,
            delay_max=settings.delay_max,
            expires_at=time.monotonic() + _PENDING_FOLLOW_UP_TTL_SECONDS,
        )
        return pending_id

    def _pop_pending_follow_up(self, pending_id: str) -> PendingFollowUp | None:
        """检索并提取处于挂起状态下的延迟补发分段上下文。"""
        self._prune_expired_pending_follow_ups()
        return self._pending_follow_ups.pop(pending_id, None)

    def _prune_expired_pending_follow_ups(self) -> None:
        """扫描并清理所有因超时失效的延迟补发上下文缓存。"""
        if not self._pending_follow_ups:
            return
        now = time.monotonic()
        expired_keys = [
            key for key, entry in self._pending_follow_ups.items() if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._pending_follow_ups.pop(key, None)

    def _track_follow_up_task(self, task: asyncio.Task[Any]) -> None:
        """追踪正在后台运行的异步补发任务，并在其完成后清理任务句柄。"""
        self._active_follow_up_tasks.add(task)
        task.add_done_callback(self._active_follow_up_tasks.discard)

    async def _drain_tasks(self) -> None:
        """强制等待或取消所有正在进行的后台补发异步任务，防止插件卸载时出现孤儿任务。"""
        tasks = [task for task in list(self._active_follow_up_tasks) if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_follow_up_segments(self, pending: PendingFollowUp) -> None:
        """后台异步主循环，按照时间延迟算法逐条补发余下的消息分段。"""
        try:
            with self._guard_session(pending.session):
                for segment in pending.segments:
                    delay = calculate_send_delay(
                        segment,
                        pending.delay_base,
                        pending.delay_per_char,
                        pending.delay_max,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

                    sent = await self.context.send_message(
                        pending.session,
                        MessageChain([Plain(segment)]),
                    )
                    if not sent:
                        logger.error("智能分段补发失败，会话: %s", pending.session)
                        return
        except asyncio.CancelledError:
            logger.warning("智能分段后台补发任务被取消，会话: %s", pending.session)
            raise
        except Exception as exc:
            logger.error("智能分段后台补发任务异常: %s", exc, exc_info=True)

    @contextmanager
    def _guard_session(self, session: str):
        """会话事务保护锁，通过引用计数保护会话不被并发的外部逻辑打碎。"""
        normalized_session = str(session or "").strip()
        if not normalized_session:
            yield
            return

        self._send_guards[normalized_session] = self._send_guards.get(
            normalized_session,
            0,
        ) + 1
        try:
            yield
        finally:
            remaining = self._send_guards.get(normalized_session, 0) - 1
            if remaining > 0:
                self._send_guards[normalized_session] = remaining
            else:
                self._send_guards.pop(normalized_session, None)

    def _is_session_guarded(self, session: str) -> bool:
        """检查当前会话是否处于补发保护锁定状态。"""
        normalized_session = str(session or "").strip()
        return bool(normalized_session and self._send_guards.get(normalized_session, 0))
