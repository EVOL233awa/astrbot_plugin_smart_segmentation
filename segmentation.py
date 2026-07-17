"""智能分段纯逻辑工具。"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from typing import Any

# 正则表达式定义
THINKING_TAG_RE = re.compile(
    r"<thinking>(?:[\s\S]*?)(?:</thinking>|$)",
    flags=re.IGNORECASE,
)
THINKING_BOUNDARY_RE = re.compile(r"</?thinking>", flags=re.IGNORECASE)

# 协议与工具调用伪影清洗正则（已移除对通用 ``` 代码块的粗暴拦截，仅清除特定工具调用和系统标签）
PROTOCOL_ARTIFACTS_RE = re.compile(
    r"("
    r"<tool_call>[\s\S]*?(?:</tool_call>|$)"
    r"|<function_call>[\s\S]*?(?:</function_call>|$)"
    r"|\[Tooltip:[^\]]*(?:\]|$)"
    r"|\[System:[^\]]*(?:\]|$)"
    r")",
    flags=re.IGNORECASE,
)

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


def is_ascii_alnum(char: str) -> bool:
    """仅判定字符是否为 ASCII 范围内的字母或数字，避免中文字符干扰。"""
    return char.isascii() and char.isalnum()


# 颜文字与纯标点识别器
def is_kaomoji_or_pure_punct(text: str) -> bool:
    """判定文本是否为纯标点、连续标点或短颜文字，用于防止其孤立成段。"""
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
        
    # 1. 含有中文汉字则绝不判定为纯颜文字，防止语义短句被强行向前合并
    if re.search(r"[\u4e00-\u9fa5]", text):
        return False
        
    # 2. 纯标点检测
    puncts = set("。！？!?~.……,，、;；:：'\"()（）【】[]{}<>《》 ")
    if all(c in puncts for c in text):
        return True
        
    # 3. 常见颜文字特征检测 (排除了中文后，再检测连续符号)
    if len(text) <= 10 and re.search(r"[><☆✨ㅠ^°¯\*_\-\|/\\]{2,}", text):
        return True
        
    # 4. 经典字母颜文字白名单
    if text.lower() in ["qwq", "awa", "qaq", "ovo", "uwu", "orz", "t_t", "q_q", "x_x", "tv_t"]:
        return True
        
    return False


# 基础清洗与检测工具
def strip_thinking_content(text: str) -> str:
    """移除 thinking 标签、Tool Call 伪影，只保留最终可见正文（保留通用代码块）。"""
    if not text:
        return ""
    cleaned_text = THINKING_TAG_RE.sub("", str(text))
    cleaned_text = THINKING_BOUNDARY_RE.sub("", cleaned_text)
    cleaned_text = PROTOCOL_ARTIFACTS_RE.sub("", cleaned_text)
    return cleaned_text.strip()


def is_non_natural_language(text: str) -> bool:
    """检测文本是否为完整的 JSON、纯代码等非自然语言。"""
    if not text:
        return False
    stripped = text.strip()
    
    # 1. 优先全局守卫：汉字占比超 10% 绝不判定为非自然语言，防止结构化文档退化为整条发送
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fa5]", stripped))
    if len(stripped) > 0 and (chinese_char_count / len(stripped)) >= 0.10:
        return False
        
    # 2. 结构化 JSON 解析校验
    if (stripped.startswith("{") and stripped.endswith("}")) or \
       (stripped.startswith("[") and stripped.endswith("]")):
        try:
            json.loads(stripped)
            return True
        except Exception:
            pass
            
    # 3. 纯代码块特征保护
    if "```" in stripped:
        return True
        
    # 4. 强特征配置字段
    if '"mcpServers"' in stripped or '"active":' in stripped or '"command":' in stripped:
        return True
        
    # 5. 特殊字符密度综合判定（已移除多余的二次中文校验）
    special_chars = set('{}[]":;,=<>')
    special_count = sum(1 for char in stripped if char in special_chars)
    if len(stripped) > 20 and (special_count / len(stripped)) > 0.15:
        return True
        
    return False


def safe_escape_segments(segments: list[str]) -> list[str]:
    """
    高选择性安全后处理器：
    只对自然语言段落中的半角尖括号进行转义，严格绕过代码块、JSON、Markdown 语法标记，
    在保障原本特性的前提下，完美解决尖括号物理吞噬问题。
    """
    processed_segments = []
    for seg in segments:
        stripped = seg.strip()
        
        # 1. 如果是非自然语言段（代码块、纯 JSON、配置字段等），严格保持原文原样
        if is_non_natural_language(seg) or "```" in stripped:
            processed_segments.append(seg)
            continue
            
        # 2. 如果是 Markdown 块引用（以 > 开头），保留开头的 Markdown 语法标记，仅转义后续文本内容
        if stripped.startswith(">"):
            boundary = seg.find(">")
            header = seg[:boundary + 1]
            body = seg[boundary + 1:]
            escaped_body = body.replace("<", "＜").replace(">", "＞")
            processed_segments.append(header + escaped_body)
            continue
            
        # 3. 针对常规自然语言（如带有颜文字的消息段），执行安全替换
        escaped_seg = seg.replace("<", "＜").replace(">", "＞")
        processed_segments.append(escaped_seg)
        
    return processed_segments


def is_markdown_heavy(text: str) -> bool:
    """
    检测文本是否包含大量 Markdown 格式特征。
    其判定结果供调用方决策是否放弃句末标点切分，改用双换行段落拆分，以保护内部格式。
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
        
    md_patterns = [
        r"^#{1,6}\s", r"^[\*\-\+]\s", r"^\d+\.\s", r"^>\s",
        r"^`{3}", r"^\|.*\|", r"^[-*_]{3,}$",
    ]
    md_line_count = 0
    for line in lines:
        stripped = line.strip()
        for pattern in md_patterns:
            if re.match(pattern, stripped):
                md_line_count += 1
                break
                
    if len(lines) > 0 and (md_line_count / len(lines)) > 0.20:
        return True
    if md_line_count >= 2:
        return True
        
    return False


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


def robust_json_loads(json_str: str) -> Any:
    """针对小模型进行高容错率的 JSON 解析，利用 ast.literal_eval 兼容单引号。"""
    s = json_str.strip()
    if not s:
        raise ValueError("Empty string")
        
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
        
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
        
    s = re.sub(r",\s*]$", "]", s)
    return json.loads(s)


def normalize_segments(segments: Any, max_segments: int | str = 0) -> list[str]:
    """标准化并自动扁平化段落。使用空格拼接防止文本粘连。"""
    try:
        max_segs = int(max_segments) if max_segments is not None else 0
    except (ValueError, TypeError):
        max_segs = 0
        
    if not isinstance(segments, list):
        if isinstance(segments, str):
            return [segments.strip()] if segments.strip() else []
        return []
        
    result: list[str] = []
    
    def flatten(item: Any):
        if isinstance(item, list):
            for sub_item in item:
                flatten(sub_item)
        elif item is not None:
            item_str = str(item).strip()
            if item_str:
                result.append(item_str)
                
    flatten(segments)
    
    if max_segs > 0 and len(result) > max_segs:
        head = result[: max_segs - 1]
        tail = " ".join(result[max_segs - 1 :])
        result = head + [tail]
        
    return result


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
    """合并或自愈括号平衡，防止因模型丢括号导致全局粘连合并。"""
    if not segments:
        return list(segments)
        
    fixed_segments = []
    for segment in segments:
        s = segment
        for open_bracket, close_bracket in BRACKET_PAIRS:
            open_count = s.count(open_bracket)
            close_count = s.count(close_bracket)
            if open_count > close_count:
                s += close_bracket * (open_count - close_count)
            elif close_count > open_count:
                s = open_bracket * (close_count - open_count) + s
        fixed_segments.append(s)
        
    return fixed_segments


def split_text_at_brackets(text: str) -> list[str]:
    """把单段文本按括号边界拆成片段。已修复中文括号前置字符导致的失效问题。"""
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
                # 修复中文失效：仅在 ASCII 字母数字、点、下划线后，才判定为程序函数调用并不予切分
                if index > 0 and (is_ascii_alnum(text[index - 1]) or text[index - 1] in "._"):
                    break
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
    max_segments: int | str,
) -> list[str]:
    """把每段内的括号包裹内容拆成独立的消息段。采用强类型防御机制。"""
    try:
        max_segs = int(max_segments) if max_segments is not None else 0
    except (ValueError, TypeError):
        max_segs = 0
        
    if not segments:
        return list(segments)
        
    result: list[str] = []
    for segment in segments:
        for part in split_text_at_brackets(segment):
            stripped_part = part.strip()
            if stripped_part and not re.match(r"^[_\-\*\s]{3,}$", stripped_part):
                result.append(stripped_part)
                
    if not result:
        return result
        
    if max_segs > 0 and len(result) > max_segs:
        head = result[: max_segs - 1]
        tail = " ".join(result[max_segs - 1 :])
        result = head + [tail]
        
    return result


def normalize_response_text_for_key(text: str) -> str:
    """归一化用于查找预分段缓存的文本。"""
    cleaned = strip_thinking_content(str(text or ""))
    if not cleaned:
        return ""
    return " ".join(cleaned.split())


def hash_normalized_text(text: str) -> str:
    """对归一化后的文本做稳定哈希。"""
    normalized = normalize_response_text_for_key(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


# Prompt 工程防御
def build_segmentation_prompt(text: str, style: str, max_segments: int) -> str:
    """构建智能分段提示词。"""
    style_guide = STYLE_GUIDES.get(style, STYLE_GUIDES["natural"])
    return f"""你是一个聊天消息切分助手。请把输入的完整文本切分成适合微信分条发送的消息。
【聊天风格】
{style_guide}
【切分方法】
请将输入文本中的所有换行（回车）全部替换为分割符号 " | "（即空格、竖线、空格），输出为单行文本。
【强制规则】
绝对不能改写、遗漏或补充原文中的任何文字和括号，必须保留原文的所有字词。
绝对不能换行！输出中绝对不允许包含 \n 或任何实际换行，必须整合成一整行纯文本。
去掉每条消息末尾的句号。
[核心保护] 绝对禁止将连续的标点符号（如“？！”、“...”、“。~”）或颜文字（如“><”、“T_T”、“qwq”、“(><)”）从主句中切开！它们必须与前文保持在同一条消息中。
【示例 1：短日常对话】
输入：
（开心）今天发工资了。
要不要一起去吃大餐？
输出：
（开心） | 今天发工资了 | 要不要一起去吃大餐
【示例 2：包含颜文字与连续标点】
输入：
欸？！><
今天天气真好...qwq
输出：
欸？！>< | 今天天气真好...qwq
请对以下文本进行分段处理：
{text}"""


# 核心解析与兜底逻辑
def parse_segments_from_model_output(
    raw_text: str,
    *,
    fallback_text: str,
    max_segments: int | str,
) -> list[str]:
    """解析模型分段输出。强类型防御机制包装。"""
    try:
        max_segs = int(max_segments) if max_segments is not None else 0
    except (ValueError, TypeError):
        max_segs = 0
        
    try:
        cleaned = strip_thinking_content(raw_text).strip()
        cleaned = cleaned.replace("(", "（").replace(")", "）")
        cleaned = cleaned.replace("[", "【").replace("]", "】")
        
        if "|" in cleaned:
            raw_segs = re.split(r"\s*\|\s*", cleaned)
            segments = []
            for seg in raw_segs:
                for line in seg.splitlines():
                    stripped_line = line.strip()
                    if stripped_line:
                        segments.append(stripped_line)
        elif cleaned.startswith("[") and cleaned.endswith("]"):
            json_text = extract_json_array_text(cleaned)
            segments = robust_json_loads(json_text)
        else:
            segments = [line.strip() for line in cleaned.splitlines() if line.strip()]
            
        normalized = normalize_segments(segments, max_segments=max_segs)
        
        # 防御大模型切碎颜文字的后处理缝合
        final_segs = []
        for seg in normalized:
            if final_segs and is_kaomoji_or_pure_punct(seg):
                final_segs[-1] += seg
            else:
                final_segs.append(seg)
        normalized = final_segs
        
        # 复读 Few-shot 幻觉检测与降级机制
        joined_result = "".join(normalized)
        hallucination_keywords = ["今天发工资了", "要不要一起去吃大餐", "今天天气真好", "公园散步", "突然下雨了", "真是不走运"]
        if any(kw in joined_result and kw not in fallback_text for kw in hallucination_keywords):
            raise ValueError("Detected few-shot hallucination")
            
    except Exception:
        fallback = strip_thinking_content(fallback_text).strip()
        return local_fallback_split(fallback, max_segs)
        
    balanced = merge_segments_balancing_brackets(normalized)
    split_result = split_segments_at_bracket_boundaries(
        balanced,
        max_segments=max_segs,
    )
    
    # 防止大模型或括号切分将颜文字剥离
    final_merged = []
    for seg in split_result:
        if final_merged and is_kaomoji_or_pure_punct(seg):
            final_merged[-1] += seg
        else:
            final_merged.append(seg)
            
    return safe_escape_segments(final_merged)


def calculate_send_delay(
    segment: str,
    delay_base: float | None,
    delay_per_char: float | None,
    delay_max: float | None,
) -> float:
    """根据文本长度计算分条发送间隔。"""
    base = float(delay_base) if delay_base is not None else 2.0
    per_char = float(delay_per_char) if delay_per_char is not None else 0.05
    max_d = float(delay_max) if delay_max is not None else 3.0
    seg_len = len(segment) if segment else 0
    normalized_delay = base + seg_len * per_char
    normalized_delay += random.uniform(0.0, 0.15)
    return max(0.0, min(max_d, normalized_delay))


def local_fallback_split(raw_text: str, max_segs: int) -> list[str]:
    """终极自愈兜底：本地纯逻辑高精切分（已优化 Markdown 笔记与非自然语言守卫顺序）。"""
    if not raw_text:
        return []
        
    # 1. 先检测整体是否为 Markdown 笔记，按段落（双换行）切分，保护内部格式
    # 调整后，即便 Markdown 中嵌有 JSON/配置/代码块，也不会被错误识别成纯“非自然语言”跳过 Markdown 段落拆分
    if is_markdown_heavy(raw_text):
        paragraphs = re.split(r'\n[ \t]*\n', raw_text.strip())
        segments = [p.strip() for p in paragraphs if p.strip()]
        if not segments:
            return safe_escape_segments([raw_text.strip()])
        if max_segs > 0 and len(segments) > max_segs:
            head = segments[: max_segs - 1]
            tail = "\n\n".join(segments[max_segs - 1 :])
            segments = head + [tail]
        return safe_escape_segments(segments)
        
    # 2. 整体非自然语言检测 (纯 JSON/纯代码)，直接整段发送 (已强化 json 解析和低汉字密度过滤校验)
    if is_non_natural_language(raw_text):
        return [raw_text.strip()]
        
    # 常规自然语言切分 (包含预处理代码块/JSON块打包)
    raw_lines = [line for line in raw_text.splitlines() if line.strip()]
    merged_lines = []
    buffer = []
    in_block = False
    block_type = None
    
    for line in raw_lines:
        stripped = line.strip()
        if not in_block:
            if stripped.startswith("```"):
                in_block = True
                block_type = 'code'
                buffer.append(line)
            elif stripped.startswith("{") or stripped.startswith("["):
                in_block = True
                block_type = 'json'
                buffer.append(line)
            else:
                merged_lines.append(stripped)
        else:
            buffer.append(line)
            if block_type == 'code' and stripped.endswith("```"):
                merged_lines.append("\n".join(buffer))
                buffer = []
                in_block = False
            elif block_type == 'json':
                text_buf = "\n".join(buffer)
                open_b = text_buf.count("{") + text_buf.count("[")
                close_b = text_buf.count("}") + text_buf.count("]")
                if open_b == close_b and open_b > 0:
                    merged_lines.append("\n".join(buffer))
                    buffer = []
                    in_block = False
                    
    if buffer:
        merged_lines.append("\n".join(buffer))
        
    sentence_split_lines = []
    for line in merged_lines:
        if "\n" in line or (line.startswith("{") and line.endswith("}")) or (line.startswith("[") and line.endswith("]")):
            sentence_split_lines.append(line)
            continue
            
        parts = []
        current = []
        depth = 0
        i = 0
        n = len(line)
        
        while i < n:
            char = line[i]
            if char in "（(【[":
                depth += 1
                current.append(char)
                i += 1
                continue
            elif char in "）)】]":
                depth = max(0, depth - 1)
                current.append(char)
                i += 1
                continue
                
            if depth > 0:
                current.append(char)
                i += 1
                continue
                
            if char == '.':
                dots = ''
                while i < n and line[i] == '.':
                    dots += '.'
                    i += 1
                current.append(dots)
                if i >= n or line[i].isspace():
                    parts.append("".join(current).strip())
                    current = []
                continue
                
            if char == '…':
                dots = ''
                while i < n and line[i] == '…':
                    dots += '…'
                    i += 1
                current.append(dots)
                if i >= n or line[i].isspace():
                    parts.append("".join(current).strip())
                    current = []
                continue
                
            # 标点聚类：吞噬后续连续的标点符号，防止 ？！ 被切碎
            if char in "。！？!?~":
                current.append(char)
                i += 1
                while i < n and line[i] in "。！？!?~.……,，":
                    current.append(line[i])
                    i += 1
                parts.append("".join(current).strip())
                current = []
                continue
                
            current.append(char)
            i += 1
            
        if current and "".join(current).strip():
            parts.append("".join(current).strip())
            
        sentence_split_lines.extend(parts)
        
    # 后处理智能合并：将孤立的纯标点或颜文字向前合并到上一个有效文本段
    merged_parts = []
    for p in sentence_split_lines:
        if merged_parts and is_kaomoji_or_pure_punct(p):
            merged_parts[-1] += p  # 向前粘连
        else:
            merged_parts.append(p)
            
    balanced = merge_segments_balancing_brackets(merged_parts)
    split_result = split_segments_at_bracket_boundaries(balanced, max_segments=max_segs)
    
    # 防止括号切分机制将带括号的短颜文字 (如 "(>_<)") 强行剥离成独立段
    final_merged = []
    for seg in split_result:
        if final_merged and is_kaomoji_or_pure_punct(seg):
            final_merged[-1] += seg  # 强行粘回上一段
        else:
            final_merged.append(seg)
            
    return safe_escape_segments(final_merged)


def get_segments_or_fallback(
    raw_text: str,
    cache: dict[str, list[str]] | None = None,
    hash_to_norm: dict[str, str] | None = None,
    *,
    max_segments: int | str = 0,
) -> list[str]:
    """智能/本地混合式分段分配器。"""
    try:
        max_segs = int(max_segments) if max_segments is not None else 0
    except (ValueError, TypeError):
        max_segs = 0
        
    cleaned_raw = strip_thinking_content(raw_text)
    if not cleaned_raw:
        return []
        
    norm_text = normalize_response_text_for_key(cleaned_raw)
    if not norm_text:
        return []
        
    exact_hash = hash_normalized_text(cleaned_raw)
    if cache and exact_hash in cache:
        return safe_escape_segments(cache[exact_hash])
        
    if cache and hash_to_norm:
        cache_norm_map = {}
        for h, segs in cache.items():
            if h in hash_to_norm and segs:
                cache_norm_map[hash_to_norm[h]] = segs
                
        remaining = norm_text.strip()
        assembled_segs = []
        matched_any = False
        
        while remaining:
            best_match_len = 0
            best_match_segs = None
            for norm_key, segs in cache_norm_map.items():
                if remaining.startswith(norm_key):
                    key_len = len(norm_key)
                    if key_len > best_match_len:
                        if key_len == len(remaining) or remaining[key_len] == ' ':
                            best_match_len = key_len
                            best_match_segs = segs
                            
            if best_match_segs:
                assembled_segs.extend(best_match_segs)
                remaining = remaining[best_match_len:].strip()
                matched_any = True
            else:
                break
                
        if matched_any and not remaining:
            return safe_escape_segments(assembled_segs)
            
    return local_fallback_split(cleaned_raw, max_segs)