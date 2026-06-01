"""智能分段纯逻辑工具。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

THINKING_TAG_RE = re.compile(
    r"<thinking>.*?</thinking>",
    flags=re.IGNORECASE | re.DOTALL,
)
THINKING_BOUNDARY_RE = re.compile(r"</?thinking>", flags=re.IGNORECASE)

BRACKET_PAIRS: tuple[tuple[str, str], ...] = (
    ("（", "）"),
    ("(", ")"),
    ("【", "】"),
    ("[", "]"),
)

STYLE_GUIDES = {
    "natural": "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。",
    "conservative": "偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息。",
    "active": "活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发。",
}


def strip_thinking_content(text: str) -> str:
    """移除 thinking 标签及其内容，只保留最终可见正文。"""
    if not text:
        return ""
    cleaned_text = THINKING_TAG_RE.sub("", str(text))
    cleaned_text = THINKING_BOUNDARY_RE.sub("", cleaned_text)
    return cleaned_text.strip()


def extract_json_array_text(raw_text: str) -> str:
    """从模型返回中提取 JSON 数组文本。"""
    result_text = str(raw_text or "").strip()
    if "```json" in result_text:
        return result_text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in result_text:
        return result_text.split("```", 1)[1].split("```", 1)[0].strip()

    start = result_text.find("[")
    end = result_text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return result_text[start : end + 1]
    return result_text


def is_action_only_text(text: str) -> bool:
    """判断文本是否整体被一对括号包裹。"""
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False

    for open_bracket, close_bracket in BRACKET_PAIRS:
        if not stripped.startswith(open_bracket) or not stripped.endswith(close_bracket):
            continue
        depth = 0
        for index, char in enumerate(stripped):
            if char == open_bracket:
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:
                    return index == len(stripped) - 1
    return False


def has_unbalanced_brackets(text: str) -> bool:
    """判断文本中是否存在未闭合的括号。"""
    for open_bracket, close_bracket in BRACKET_PAIRS:
        if text.count(open_bracket) != text.count(close_bracket):
            return True
    return False


def merge_segments_balancing_brackets(segments: list[str]) -> list[str]:
    """合并被模型拆到不同段的括号对，保证每段括号数量平衡。"""
    if not segments:
        return list(segments)

    merged: list[str] = []
    buffer = ""
    for segment in segments:
        buffer = buffer + segment if buffer else segment
        if not has_unbalanced_brackets(buffer):
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    return merged


def split_text_at_brackets(text: str) -> list[str]:
    """把单段文本按括号边界拆成片段，括号块本身作为独立片段保留。"""
    if not text:
        return []

    parts: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        matched_pair: tuple[str, str] | None = None
        for open_bracket, close_bracket in BRACKET_PAIRS:
            if char == open_bracket:
                matched_pair = (open_bracket, close_bracket)
                break

        if matched_pair is None:
            buffer.append(char)
            index += 1
            continue

        open_bracket, close_bracket = matched_pair
        depth = 1
        scan_index = index + 1
        while scan_index < len(text) and depth > 0:
            if text[scan_index] == open_bracket:
                depth += 1
            elif text[scan_index] == close_bracket:
                depth -= 1
            scan_index += 1

        if depth != 0:
            buffer.append(text[index:])
            index = len(text)
            break

        if buffer:
            parts.append("".join(buffer))
            buffer = []
        parts.append(text[index:scan_index])
        index = scan_index

    if buffer:
        parts.append("".join(buffer))

    return parts


def split_segments_at_bracket_boundaries(
    segments: list[str],
    *,
    max_segments: int,
) -> list[str]:
    """把每段内的括号包裹内容拆成独立的消息段。"""
    if not segments:
        return list(segments)

    result: list[str] = []
    for segment in segments:
        for part in split_text_at_brackets(segment):
            stripped_part = part.strip()
            if stripped_part:
                result.append(stripped_part)

    if not result:
        return result

    if max_segments > 0 and len(result) > max_segments:
        head = result[: max_segments - 1]
        tail = "".join(result[max_segments - 1 :])
        result = head + [tail]

    return result


def normalize_response_text_for_key(text: str) -> str:
    """归一化用于查找预分段缓存的文本：剥 thinking 并折叠所有空白。"""
    cleaned = strip_thinking_content(str(text or ""))
    if not cleaned:
        return ""
    return " ".join(cleaned.split())


def hash_normalized_text(text: str) -> str:
    """对归一化后的文本做稳定哈希；空文本返回空串。"""
    normalized = normalize_response_text_for_key(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def normalize_segments(segments: Any, *, max_segments: int) -> list[str]:
    """规范化模型返回的分段结果。"""
    if not isinstance(segments, list):
        raise ValueError("模型返回的分段结果不是列表")

    normalized_segments = [
        str(segment).strip() for segment in segments if str(segment).strip()
    ]
    if not normalized_segments:
        raise ValueError("模型返回的分段结果为空")

    if max_segments > 0 and len(normalized_segments) > max_segments:
        head = normalized_segments[: max_segments - 1]
        tail = "".join(normalized_segments[max_segments - 1 :])
        normalized_segments = head + [tail]

    return normalized_segments


def build_segmentation_prompt(text: str, style: str, max_segments: int) -> str:
    """构建智能分段提示词。"""
    style_guide = STYLE_GUIDES.get(style, STYLE_GUIDES["natural"])
    return f"""你正在模拟一个人用手机聊天。下面是 ta 想说的内容，请把它分成几条消息，就像真人会怎么一条一条发出来那样。

{style_guide}

规则：
- 不要改写原意，不要补充新信息
- 去掉每条消息末尾的句号「。」
- 保留感叹号、问号、省略号、波浪号等有情绪的标点
- 不要每个逗号都拆开，相关的内容放在一条里
- 消息长短可以不均匀
- 括号（中文「（）」「【】」或英文「()」「[]」）内的内容（动作、神态、旁白等描述）必须作为独立的一条消息单独发送，不要和括号外的正文合在同一条
- 括号内的内容本身不能再拆开，需保持完整
- 如果整段内容就是被括号包裹的动作/神态描述，直接整段返回不再切分
- 最多分成 {max_segments} 条
- 如果不适合切分，就返回只包含原文的一项数组

原文：{text}

只返回 JSON 数组，如 ["消息1", "消息2"]"""


def parse_segments_from_model_output(
    raw_text: str,
    *,
    fallback_text: str,
    max_segments: int,
) -> list[str]:
    """解析模型分段输出；异常时回退为原文单段。"""
    try:
        json_text = extract_json_array_text(raw_text)
        segments = json.loads(json_text)
        normalized = normalize_segments(segments, max_segments=max_segments)
    except Exception:
        fallback = strip_thinking_content(fallback_text).strip()
        return [fallback] if fallback else []

    balanced = merge_segments_balancing_brackets(normalized)
    return split_segments_at_bracket_boundaries(
        balanced,
        max_segments=max_segments,
    )


def calculate_send_delay(
    segment: str,
    delay_base: float,
    delay_per_char: float,
    delay_max: float,
) -> float:
    """根据文本长度计算分条发送间隔。"""
    normalized_delay = delay_base + len(segment) * delay_per_char
    normalized_delay += random.uniform(0.0, 0.15)
    return max(0.0, min(delay_max, normalized_delay))
