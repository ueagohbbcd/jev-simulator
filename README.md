# Jev 模拟器

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Chat Completions](https://img.shields.io/badge/backend-Chat_Completions-475569)

**用普通 Chat API，提供可配置的 Jev 兼容决策端点。**

把 Chat Completions API 封装成类型化决策服务。业务侧传入状态和问题，获得判断、分类或评分；服务侧通过一份 TOML 配置输出读取模式、提示词、比较方式和概率处理。默认读取首 token `logprobs`，也可选择模型直接报告数值概率。

[快速启动](#启动) · [配置指南](docs/configuration.md) · [API 兼容性](docs/api.md) · [概率与诊断](docs/probabilities.md)

```text
状态 + 问题 → Jev 模拟器 → Chat Completions API
                  ↑                 ↓
               TOML 配置       标签概率 / 数值报告
                  ↓                 ↓
           Noul / Choice / Score ← 解析与聚合
```

| 能力 | 用途 |
| --- | --- |
| Noul / Choice / Score | 是非判断、候选分类、分档评分 |
| 可选图片输入 | 兼容端点直接携带 base64／data URL 图片，转发给视觉上游 |
| 输出读取模式 | 标签 logprobs，或直接报告 JSON 数值概率 |
| 文本提示词模板 | 直接编辑措辞、布局和证据位置 |
| 双循环与自定义标签 | 比较候选的两种顺序，或替换默认字母标签 |
| 温度缩放 | 调整输出概率的尖锐程度 |
| 概率质量诊断 | 查看归一化前质量、缺失标签和质量上下界 |
| stdin 热重载 | 运行中切换完整配置，在途请求保持原配置 |

无需 GPU、SGLang、完整词表或选定-token 查询接口。默认模式每个比较分支只请求一个输出 token；直接报告模式生成完整 JSON，不要求 logprobs。安装后可独立运行，无前端构建步骤。

提供 Jev 风格的 Noul、Choice、Score 请求与响应，判断由配置的上游模型完成。接口说明见 [API 兼容性](docs/api.md)。

需要每次请求携带推理配置时，使用 `POST /v1/evaluate`；加 `dry_run: true` 可直接预览。见 [扩展端点](docs/api.md#扩展端点) 和 [请求示例](examples/evaluate.json)，无需预先注册配置。

使用视觉上游时，在服务配置中设置 `upstream.supports_images = true`，并在原请求中添加可选 `images`。见 [图片格式与限制](docs/api.md#图片输入) 和 [内含小图片的请求示例](examples/image-request.json)。

## 启动

需要 Python 3.11 或更新版本。

```sh
git clone https://github.com/ueagohbbcd/jev-simulator.git
cd jev-simulator
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 用 .venv\Scripts\Activate.ps1
python -m pip install -e .
```

把上游密钥放进环境变量。PowerShell：

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
jev-simulator check --config config.toml
jev-simulator serve --config config.toml
```

macOS / Linux 用 `export DEEPSEEK_API_KEY='你的密钥'`。示例默认连接 DeepSeek，并显式禁用思考；换供应商时修改 `upstream`，删除对方不支持的 `extra_body` 字段。默认模式要求首输出 token 的 logprobs；直接报告模式要求 JSON 输出。

服务默认监听 `127.0.0.1:8080`，打开 `/docs` 可交互查看和调用 HTTP 接口。

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

PowerShell 可用 `curl.exe`，把命令写在一行。业务客户端的 base URL 指向本服务，model 使用 `jev-latest` 或配置中的上游模型名。一个服务实例使用一份配置，不根据请求动态切换上游账户。

## 配置与请求入口

| 入口 | 推理配置 | 返回 |
| --- | --- | --- |
| `POST /v1/systemone` | 服务当前 TOML | 兼容答案 |
| `POST /v1/evaluate` | 请求内完整 `execution`；省略字段取代码默认值 | 有效配置、告警与兼容答案；`dry_run` 返回调用计划 |

扩展端点不需要注册 profile，不保存请求配置。上游账户、模型连接和服务容量仍由部署端管理。两个入口接受同一套问题和可选图片，详见 [API](docs/api.md)。

[config.toml](config.toml) 是完整起点。可编辑提示词、选择读取模式，组合双循环、呼号和概率后处理温度。Noul 固定一次 Yes/No 判断；双循环只用于 Choice/Score。生成温度与概率后处理温度分别配置，依赖与限制见 [配置指南](docs/configuration.md)。

以下配置均可直接通过 `--config` 使用：

- [直接报告概率](examples/reported-probability.toml)：生成 JSON 数值分布，不要求 logprobs。
- [温度缩放](examples/calibrated.toml)、[呼号](examples/callsigns.toml)、[双循环](examples/round-robin.toml)。

运行中可在服务终端输入 `status`、`reload` 或 `reload PATH`，整体替换配置；在途请求保持原快照，server 设置变更需重启。每个响应的 `x-jev-config-id` 标识有效配置。管道控制方式见 [选择与重载](docs/configuration.md#选择与重载)。

## 命令行调试

```sh
# 离线检查，不需要密钥，不发送请求
jev-simulator check --config config.toml

# 展开全部实际提示词和映射，查看双循环会发多少请求
jev-simulator preview --config config.toml --request examples/request.json

# 真实调用；stdout 是 Jev 响应，可选保存完整诊断
jev-simulator evaluate --config config.toml --request examples/request.json \
  --diagnostics diagnostics/example.json
```

预览、命令行运行和 HTTP 服务共用同一个计划、解析和聚合实现。完整诊断包含提示词、原始概率及映射，可能含输入隐私；只在明确指定路径时写入，本仓库默认忽略 `diagnostics/`。

HTTP 进程把结构化事件写到 **stderr**，包含请求 ID、配置 ID、模式和告警；logprobs 模式另有概率质量和缺失标签数量。

## 文档

- [接口与兼容性](docs/api.md)：端点、图片输入、返回与错误。
- [配置指南](docs/configuration.md)：字段、依赖关系与热重载。
- [概率与诊断](docs/probabilities.md)：数值语义、温度、双循环与质量边界。
- [优化方向](docs/direction.md)：当前实现与待实验候选。
- [验证记录](docs/verification.md)：离线覆盖与真实调用的验证范围。
- [历史提示词对照](docs/prompt-comparison.md)：一次固定题集上的对照及其限制。

## 开发

```sh
python -m pip install -e ".[test]"
python -m pytest
```

测试使用本地模拟上游，无需 API 密钥，不产生模型调用费用。

项目参考 [Jev HTTP API](https://docs.typesafe.ai/api) 和 [openjev-sglang](https://github.com/ekzhang/openjev-sglang) 的接口思路，使用普通 Chat API 读取标签概率或模型直接报告的数值。
