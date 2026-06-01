# AstrBot 智能分段插件

在主 LLM 回复生成后先调用一个分段模型，把完整回复拆成更像真人聊天的多条消息；发送前只保留首段，首段发送完成后再按延迟异步补发剩余段。

## 特性

- 使用 AstrBot 原生 hooks：`on_llm_response`、`on_decorating_result`、`after_message_sent`。
- 只处理 LLM/Agent 的纯文本结果，不改图片、语音、文件等非文本消息链。
- 支持括号动作/神态描述独立成段，并避免整段动作文本被继续拆分。
- 分段模型输出异常、超时或只返回单段时，会自动回退为原回复。

## 配置

在 AstrBot 插件 WebUI 中配置：

| 配置项 | 说明 | 默认值 |
| --- | --- | --- |
| `enabled` | 启用/禁用插件 | `true` |
| `provider_id` | 分段模型 Provider ID，留空使用当前会话模型 | `""` |
| `style` | `natural` / `conservative` / `active` | `natural` |
| `min_length` | 文本短于该长度不分段 | `15` |
| `max_segments` | 单次最多分段数量 | `8` |
| `temperature` | 分段模型温度 | `0.3` |
| `max_tokens` | 分段模型最大输出 token | `600` |
| `timeout_seconds` | 分段模型超时时间 | `12.0` |
| `delay_base` | 补发基础延迟（秒） | `0.35` |
| `delay_per_char` | 按字符增加的延迟（秒/字符） | `0.015` |
| `delay_max` | 单段最大补发延迟（秒） | `1.2` |

建议关闭 AstrBot 内置 `platform_settings.segmented_reply.enable`，避免和本插件重复分段。

## 许可证

GPL-3.0-or-later