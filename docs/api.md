# 接口与兼容性

接口依据：2026-09-20 的 [TypeSafe/Jev API 文档](https://docs.typesafe.ai/api) 和 [openjev-sglang](https://github.com/ekzhang/openjev-sglang)。本项目使用 Chat Completions API 作为推理后端。

## 核心端点

`POST /v1/systemone` 接收 `model`、`state`、`questions`。`state` 和每个问题的 `instructions` 接受字符串、对象或数组。问题 ID 原样对应响应 ID，不参与提示词推理。

| 问题 | criteria | 响应 |
| --- | --- | --- |
| Noul | 可省略；true/false 描述，各可为字符串、对象或数组 | `{type: "noul", noul: P(Yes)}` |
| Choice | 选项键到描述的对象；null 描述回退到键 | `type`、`choice`、`probabilities`、`confidence` |
| Score | 有序描述数组，2–10 个等级 | `type`、期望分 `score`、完整 `legend`、`probabilities`、`confidence` |

响应顶层是 `model`、`answers`、`usage`。`model` 保留调用方使用的受支持名称。`usage.input_tokens`、`usage.output_tokens` 汇总实际上游用量，包括多问题和双循环；输入 token 包含供应商报告的缓存输入。

Score 等级从 0 开始，结果可以是小数。结构化等级描述在 legend 中序列化为 JSON 字符串。Choice 和 Score 的 confidence 使用 `1 - H(p)/log(n)`，反映分布集中程度，取值为 0–1。这是本实现采用的计算公式。

每个 HTTP 响应通过 `x-typesafe-request-id` 关联日志，`x-jev-config-id` 标识有效配置。诊断不混入标准 answers。

## 图片输入

`/v1/systemone` 顶层增加可选 `images`，`/v1/evaluate` 放在 `request.images`。省略或空数组保持纯文本行为和原响应结构；这是模拟器的扩展字段，不保证其他 System One 服务接受。

```json
"images": [
  {"data": "data:image/png;base64,..."},
  {"type": "image/jpeg", "data": "<base64 图片字节>"}
]
```

支持 JPEG、PNG、WebP、静态 GIF。只接受内联图片内容，不读取本地路径或远程 URL。图片按顺序加入每个调用的 user 消息，位于模板渲染的文本之前；上游收到标准 Chat Completions `image_url` 内容块。普通 state 对象继续作为文本，不解释其中的图片链接。双循环的每个分支都会携带图片，因此图片输入用量也会随调用次数增加。

部署须设置 `upstream.supports_images = true`，表示已选择支持图片的上游；这不是自动能力探测。默认关闭，带图片请求返回 422。两种读取模式都能携带图片，上游还须支持该模式要求的 logprobs 或 JSON 输出。

最多 8 张，每张解码后的上传文件不超过 12 MiB、合计 32 MiB，每张不超过 1600 万像素。服务校验内容与声明格式，拒绝动画和损坏图片，应用 EXIF 方向并去除元数据，在内存中转为 PNG；转换后合计也不得超过 32 MiB，不落临时文件。请求体仍受 `server.max_body_bytes` 限制，默认 2 MiB（包含 base64）；较大图片需由部署端提高，例如 48 MiB。

[image-request.json](../examples/image-request.json) 内含一张小红色色块，可直接用于兼容端点。预览与完整诊断包含转换后的图片数据；普通日志与校验错误不包含图片内容。`dry_run` 会执行本地图片校验与转换，但不调用上游。

## 扩展端点

`POST /v1/evaluate` 接收 `{request, execution, dry_run}`。`request` 为上述 System One 请求；必填的 `execution` 是本次推理配置，仅允许 `adapter`、`prompt`、`generation`、`diagnostics`。字段说明与 TOML 相同，省略字段取代码默认值，不继承运行中服务的推理设置。`execution: {}` 表示代码默认推理配置。

上游连接、模型、密钥、供应商扩展参数和服务容量由进入请求时的服务快照提供，不能通过 execution 覆盖。模型名称仍按兼容端点规则校验。没有 profile 名称、配置注册或会话状态；本次配置不会改动服务默认值。

```json
{
  "request": {
    "model": "jev-latest",
    "state": "一枚公平硬币已抛出，结果未观察。",
    "questions": {"heads": {"type": "noul", "instructions": "结果是正面吗？"}}
  },
  "execution": {
    "adapter": {"mode": "reported_probability"},
    "prompt": {"system": "Report probabilities using the evidence. Return only the requested JSON."}
  },
  "dry_run": true
}
```

`dry_run` 默认 false。true 时不调用上游，返回有效 `execution`、`config_id`、`warnings` 和 `plan`（`request_count` 与展开后的 `requests`，含实际提示词和映射）。false 时返回相同配置元数据及 `result`，后者是完整兼容答案。两种路径共用配置校验与调用计划；预览不保证上游在线、支持该协议或模型输出有效。

完整示例见 [evaluate.json](../examples/evaluate.json)。向此端点发送该文件可预览，改为 `dry_run: false` 执行。两种入口共用鉴权、并发池、调用数及请求大小限制。`/v1/limits` 的候选容量描述服务默认配置；本次呼号产生的候选限制由本次计划校验。

非法组合返回 422 及字段位置；合法但不适用的设置以 `warnings` 返回。例如全为 Noul 时开启双循环或自定义呼号，以及直接报告模式显式设置 token 质量阈值。不会自动开启依赖项或修改提示词。

成功响应正文 `config_id` 与响应头 `x-jev-config-id` 一致，包含本次推理配置和服务端运行配置的摘要。解析出有效配置之后的失败也使用该 ID；配置本身非法时仍使用进入请求时的服务配置 ID。响应不暴露服务端连接或密钥配置。预览包含本次输入和提示词，不写入普通服务日志。

## 辅助端点

| 路由 | 用途 |
| --- | --- |
| `/v1/models` | 支持 `jev-latest` 和配置模型名；同时提供 TypeSafe 与 OpenAI 风格列表 |
| `/v1/limits` | 本实例真实的题数、候选数、字节数、调用数和并发上限 |
| `/health/live` | 进程存活 |
| `/health` | 检查本地配置及密钥是否就绪；上游连通性可通过 `evaluate` 验证 |
| `/docs`、`/openapi.json` | 交互文档和接口结构 |
| `/` | 服务信息 |

## 错误

请求字段不合法、未知模型、候选数或展开后的调用数超限，都在推理前返回 422。过大的请求体返回 413；本服务鉴权失败返回 401；批次容量已满返回 529；上游限流返回 429；超时返回 504；上游认证、无有效概率等失败返回网关错误。错误消息提供状态与原因摘要。

成功响应包含全部问题的结果。任一问题失败时，整个批次返回错误，剩余任务取消。重试由调用方发起，并作为新请求计费。

## 与官方服务、SGLang 版本的差异

1. **概率获取方式**：默认读取 top-k 及实际采样 token。缺失目标按 0 近似；诊断提供目标质量上下界和缺失列表。目标全缺失时返回错误。直接报告模式读取 JSON 数值分布，不要求 logprobs；非法分布返回网关错误。
2. **候选容量**：默认 A–Z，最多 26 个候选；非空 callsigns 列表决定实际容量，最多 255。Score 最多 10 级。候选数较多时，需结合缺失标签诊断评估 top-k 的覆盖情况。`/v1/limits` 公布本实例容量。
3. **证据布局**：聊天记录完整序列化为证据，消息角色与布局由提示词模板决定。
4. **数值语义**：结果取决于上游模型、提示词和温度；双循环输出归一化锦标赛胜分。Choice 平局时按输入选项顺序选取。
5. **输入预算**：本地限制请求字节和展开调用数；上下文 token 上限由上游服务检查。

配置加载与推理核心也可作为 Python 模块使用。
