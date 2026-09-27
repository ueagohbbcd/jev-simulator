# 配置指南

一个进程使用一份完整 TOML，通过 `--config` 选择启动文件，也可以在运行中用 stdin 命令替换。兼容端点 `POST /v1/systemone` 按这份配置执行，调用方只需提交状态、问题和可选图片。`jev-simulator check --config PATH` 校验本地配置；`jev-simulator evaluate` 用于从命令行实际调用上游。

需要逐请求选择推理方法时，可通过可选的扩展端点 [`POST /v1/evaluate`](api.md#扩展端点) 的 `execution` 携带本次 `adapter/prompt/generation/diagnostics`，使用同一套字段和依赖校验。省略字段使用代码默认值，不合并服务推理配置；不会修改 TOML 或影响其他请求。上游连接与服务设置仍由部署管理。`dry_run: true` 返回有效配置和调用计划。

## 选择与重载

在运行 `jev-simulator serve --config config.toml` 的终端输入 `status`、`reload` 或 `reload PATH`。路径有空格时可以用一对引号包围；Windows 反斜杠不作为转义符。相对路径基于启动工作目录，不基于上一份配置所在目录。

`reload` 在本地加载和校验整份文件，检查上游密钥环境变量是否存在，再切换当前配置快照。文件读取失败、TOML 语法错误、提示词缺占位符或需要重启的设置发生变化，都返回失败回执，继续使用原配置和路径。

可热换 `[upstream]`、`[adapter]`、`[generation]`、`[prompt]`、`[diagnostics]`；`[server]` 的所有字段固定于进程启动时，变更需重启。切换配置文件时要带上相同的非默认 server 设置，省略字段表示使用默认值，不表示继承旧值。

每次 HTTP 请求在进入服务时捕获一个快照，包括最终响应头和失败日志的配置 ID。重载不影响已开始的请求，也不重置全进程并发额度。stdout 是逐行 JSON 命令回执，stderr 是服务日志；stdin EOF 不会关闭 HTTP 服务。

其他程序可以控制它启动的服务子进程。例如，下面只查询状态和切配置，不运行推理；密钥环境变量由子进程继承：

```python
import json
import subprocess

process = subprocess.Popen(
    ["jev-simulator", "serve", "--config", "config.toml"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1,
)
try:
    for command in ["status", "reload examples/calibrated.toml"]:
        process.stdin.write(command + "\n")
        process.stdin.flush()
        print(json.loads(process.stdout.readline()))
finally:
    process.terminate()
    process.wait()
```

这段示例结束时主动停止子进程。长期控制程序可以持续持有管道；stderr 默认继承到终端，stdout 返回 JSON 回执。通过启动服务的终端或父进程发送控制命令。

## upstream：模型连接

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `base_url` | `https://api.deepseek.com` | Chat API 前缀，程序追加 `/chat/completions`；OpenAI 风格地址通常包含 `/v1` |
| `model` | `deepseek-flash` | 发给上游的实际模型名 |
| `api_key_env` | `DEEPSEEK_API_KEY` | 从哪个环境变量读取上游密钥 |
| `timeout` | `60.0` | 单次上游 HTTP 调用超时秒数 |
| `top_logprobs` | `20` | 请求返回的候选数，1–20；供应商需要支持所选值 |
| `supports_images` | `false` | 部署声明上游支持内联图片；带图片请求需开启，不做在线能力探测 |
| `extra_body` | `{}` | 供应商扩展，例如 `thinking = { type = "disabled" }` |

`token_logprobs` 模式固定一个输出 token、单条非流式 completion、开启 logprobs、采样温度 1。`reported_probability` 模式使用 `[generation]` 设置、`response_format = {type: "json_object"}`，不请求 logprobs。模型答案由概率分布计算。`extra_body` 用于供应商扩展；覆盖固定协议字段或设置工具调用、JSON 输出格式时，配置校验会报错。换供应商时通常只需换连接配置并删去 `thinking`。

上游使用 Chat Completions，并以 `upstream.api_key_env` 指定的密钥认证。

设置 `supports_images = true` 后接受可选图片，将其转换为 user 消息中的 `image_url` 内容块。须先选择支持视觉的模型；不能仅通过开关赋予模型视觉能力。此设置由服务端管理，不属于请求级 execution。较大图片可提高 `[server] max_body_bytes`，例如 `50331648`（48 MiB）；base64 数据也计入请求体。图片限制见 [API](api.md#图片输入)。

## adapter：模式与组合

`mode` 可为 `token_logprobs`（默认）或 `reported_probability`。后者读取模型生成的 JSON 标签概率，须显式提供 `[prompt].system`，不能沿用默认的单标签系统提示词。可从 [完整示例](../examples/reported-probability.toml) 开始。两种模式都支持逐分支温度变换和 Choice/Score 双循环；每题或每场比较独立调用，不进行多题共同回答。

`temperature` 为正有限数，默认 1，用于概率后处理。公式是 `softmax(label_logprobs / temperature)`，详见概率文档。

`double_round_robin` 默认 false。开启后 Choice/Score 的 n 个选项需要 `n(n-1)` 次调用；Noul 仍为 1 次。三选一是 6 次，十选一是 90 次。完整批次超过调用上限会在任何上游调用前拒绝。

`callsigns` 默认 `[]`，使用 A–Z。非空列表按顺序绑定候选，例如：

```toml
[adapter]
callsigns = ["alpha", "fox", "delta", "echo"]
```

单次四选一会使用这四个词；双循环每场只使用前两个词，两次交换候选含义。列表长度同时限制 Choice/Score 的最大候选数。每个标签须为非空、无空白的唯一字符串，按大小写精确匹配。Noul 固定使用 Yes/No。

logprobs 模式的呼号应在目标模型的首输出位置对应单个 token；直接报告模式的呼号是 JSON 键，不要求单 token。可用供应商 tokenizer 核对，并通过实际调用的诊断检查标签概率。示例候选基于公开 DeepSeek tokenizer 选取；切换模型时应重新核对分词。

## generation：直接报告模式的生成设置

`temperature` 默认 0，范围 0–2；`max_tokens` 默认 1024，为正整数。这是上游生成参数，与 `adapter.temperature` 后处理温度分开。输出长度不足时返回错误，不接受被截断的 JSON。仅 `reported_probability` 可配置此表；默认 logprobs 模式显式提供该表会报错。

## prompt：直接可读的文本

`system` 和 `user` 都是普通字符串，可用 TOML 多行字符串。两个字段合起来必须包含全部四个占位符：

| 占位符 | 渲染内容 |
| --- | --- |
| `{{state}}` | 原始字符串，或保留全部字段的 JSON |
| `{{instructions}}` | 当前问题的说明；结构化说明也转为 JSON |
| `{{options}}` | 当前调用的标签及描述；双循环只含本场两个候选 |
| `{{output}}` | 本场单标签要求，或 JSON 标签数值分布要求，由 mode 决定 |

自定义问题 ID 不发给模型；Choice 的语义键也不发给模型，除非描述是 null，此时用键作为描述。Noul 的 true/false 标准渲染在 Yes/No 后。

模板直接决定布局和重复次数。想复读问题，可再放一次 `{{instructions}}`；想复读全部，直接再写一遍对应段落。字面 JSON 的单大括号原样保留。`{{...}}` 用于已知占位符，未知或不闭合的写法会报错。替换执行一次，证据里的同名字样原样保留。

## server：运行边界

| 字段 | 默认 |
| --- | --- |
| `host` / `port` | `127.0.0.1` / `8080` |
| `api_key_env` | 省略，即本服务不要求鉴权 |
| `max_questions` | 64 |
| `max_body_bytes` | 2097152 |
| `max_calls_per_request` | 256 |
| `max_concurrent_requests` | 16 |
| `max_concurrent_calls` | 8 |
| `request_timeout` | 120.0 秒，整个批次 |

例如，配置 `api_key_env = "JEV_GATEWAY_API_KEY"`，再设置对应环境变量，就为 `/v1/*` 加上 Bearer 鉴权。这与上游 `DEEPSEEK_API_KEY` 分开。配置了密钥变量却没有设置值时，服务拒绝启动。

并发限制是单进程的。超过批次并发限制返回 529；上游 429 保留为限流信号。一次批次失败或超时会取消尚未完成的任务；上游已接收调用的执行与计费按供应商规则处理。

## diagnostics：告警门槛

`low_mass_threshold` 默认 0.99，仅用于 logprobs 模式。比较的是温度缩放与目标归一化**之前**的概率质量，低于门槛时随结果输出告警。详细判定见 [概率与诊断](probabilities.md)。
