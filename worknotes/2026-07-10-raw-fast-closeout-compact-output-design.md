# Raw-fast closeout 预览命令紧凑输出设计

日期：2026-07-10

状态：已确认（2026-07-10）

## 1. 问题与目标

`closeout_command.preview.sh` 当前直接调用 `ops.raw_fast_closeout`。该 CLI 会把完整 closeout 结果以缩进 JSON 打到标准输出，其中包含预检、证据报告、计时、清理、最终验证和 native/wiki 状态等完整子树。因此一次正常 closeout 也可能输出数百行。

本次修改的目标是：

- 生成的 `closeout_command.preview.sh` 默认使用紧凑输出。
- 成功时只给出操作员需要的 closeout 摘要，不打印内部审计树。
- 失败时只打印失败阶段及其相关错误信息，不复述其他已经成功的阶段。
- 完整诊断继续写入现有报告文件，紧凑输出只给出报告路径。
- 保持原始 CLI 的默认完整 JSON 契约，避免破坏已有脚本和测试。

## 2. 已确认根因

当前数据流如下：

1. `ops/raw_fast_ingest_prepare.py` 生成 `closeout_command.preview.sh`。
2. 预览脚本直接调用 `python -m ops.raw_fast_closeout`，没有选择输出模式。
3. `ops/raw_fast_closeout.py` 的成功和失败出口都调用缩进式 `print_json(...)`。
4. 成功 payload 同时携带 `pre_verify`、`marked`、`mark_pending`、`evidence_reports`、`final_verify`、wiki/native 状态、计时和日志等完整结构。
5. 代码中已经存在 `build_raw_fast_session_summary(...)`，但生产出口没有使用它；现有测试只单独验证该 helper 足够紧凑。

所以问题不是 closeout 执行了过多步骤，而是“面向审计的完整结果”被直接当作“面向终端的操作摘要”打印了出来。

## 3. 输出契约

### 3.1 CLI 模式

为 `ops.raw_fast_closeout` 增加：

```text
--output-mode {full,compact}
```

- 默认值为 `full`，保持直接调用者的现有完整 JSON 输出不变。
- 生成的 `closeout_command.preview.sh` 显式传入 `--output-mode compact`。
- 两种模式只影响终端渲染，不改变 closeout 步骤、报告写入、退出码或失败判定。

### 3.2 成功输出

紧凑模式复用 `build_raw_fast_session_summary(...)`，只渲染其面向操作员的 markdown 摘要。目标为不超过 8 个可见行，至少包含：

- `raw_fast_ok`
- raw note 路径
- final/evidence report 路径或数量
- wiki pending / native blocked 状态
- 临时目录是否已清理

不得包含 `pre_verify`、逐文件 evidence summaries、逐步骤 timings、完整 cleanup 明细或完整 final verify 子树。

### 3.3 失败输出

紧凑模式使用单独的失败摘要构造器，只保留：

- `raw_fast_ok: false`
- 失败 `stage`
- 顶层 `error` / `errors`（若存在）
- 该阶段内实际失败的检查项或 cleanup 条目
- `returncode` / `command_returncode`（若存在）
- `blocker_diagnostics` / `diagnostic_hint`（若存在）
- 已生成的诊断或 evidence report 路径（若存在）
- `stderr` 的尾部片段（若存在，最多 20 个非空行）

裁剪规则：

- 不打印与失败无关、且已经成功的阶段子树。
- 列表只保留明确失败或包含错误字段的条目。
- 长文本只保留尾部诊断，不回显完整子进程 transcript。
- 找不到更细粒度错误字段时，打印当前失败阶段的最小 payload，并附报告路径；不得退回打印整个 closeout payload。
- 输出精简不能吞掉非零退出码。

## 4. 设计审计

### 4.1 迭代一：直接改变 CLI 默认输出

方案：让 `ops.raw_fast_closeout` 默认只打印摘要。

结论：拒绝。现有测试和潜在脚本会解析完整 JSON；直接修改默认值会把终端降噪变成公共契约破坏。

### 4.2 迭代二：预览脚本显式选择紧凑模式

方案：保留 CLI 默认 `full`，增加显式模式，由生成的预览脚本选择 `compact`；成功、失败分别使用集中式摘要构造器。

结论：采用。该方案把行为变化限制在用户实际运行的预览命令，同时保留机器调用兼容性，并让成功/失败输出共享可测试的稳定边界。

### 4.3 基线证据台账

| ID | 当前证据 | 设计影响 |
| --- | --- | --- |
| B1 | 预览脚本直接调用 closeout CLI，没有输出选项 | 生成命令增加紧凑模式参数 |
| B2 | `print_json` 使用缩进 JSON，并被成功/失败出口共同调用 | 增加统一输出分派，不改变 full 路径 |
| B3 | compact session summary helper 已存在但未接入生产 | 成功 compact 路径复用该 helper |
| B4 | CLI 测试依赖完整嵌套 JSON | 默认 full 契约必须保持 |
| B5 | compact helper 只有单元测试，没有 CLI/预览集成测试 | 新增端到端输出测试 |

### 4.4 决策台账

| ID | 决策 | 理由 |
| --- | --- | --- |
| D1 | CLI 默认保持 `full` | 兼容现有机器调用者 |
| D2 | 预览脚本显式使用 `compact` | 只改变目标用户入口 |
| D3 | 成功复用现有 session summary | 避免第二套摘要语义 |
| D4 | 失败使用白名单式投影 | 确保只报告错误相关内容 |
| D5 | 详细信息保留在 artifact 中 | 兼顾终端可读性与审计能力 |
| D6 | 输出模式不影响退出码与执行流程 | 防止降噪掩盖失败 |

### 4.5 压缩台账

| 内容 | 终端 compact 输出 | 报告 artifact |
| --- | --- | --- |
| 成功摘要 | 保留 | 保留 |
| 失败阶段与错误 | 保留 | 保留 |
| 成功阶段的完整子树 | 删除 | 保留 |
| evidence 逐文件明细 | 删除 | 保留 |
| timings 逐步骤明细 | 删除 | 保留 |
| 完整 stderr/transcript | 删除，仅保留尾部 | 保留（若当前流程已落盘） |

## 5. 影响范围

预计只修改：

- `ops/raw_fast_closeout.py`
- `ops/raw_fast_ingest_prepare.py`
- `tests/test_raw_fast_closeout.py`
- `tests/test_raw_fast_ingest_prepare.py`

如果活跃 skill/reference 中复制了 closeout 命令，则只同步相关命令行片段及项目内镜像；不扩写 quickstart，也不把实现细节复制到多处。

## 6. 测试与验收

实现阶段按 RED-GREEN-REFACTOR 执行，至少覆盖：

1. compact 成功：退出码为 0，输出不超过 8 个可见行，不出现完整内部子树。
2. compact 预检/验证失败：保留失败阶段、错误诊断和非零退出码，不出现 cleanup、mark 或 evidence 等无关成功分支。
3. compact 清理失败：只列失败 cleanup 条目和相关诊断，不回显完整 pre-verify/control-scan 数据。
4. 默认 full：现有嵌套 JSON 字段保持可解析，原有兼容测试继续通过。
5. 预览生成：`closeout_command.preview.sh` 包含 `--output-mode compact`。
6. 生成脚本实跑：成功输出满足行数预算，失败输出满足错误投影规则。

## 7. 风险与约束

- 失败 payload 的形状随 stage 不同，摘要构造器必须有确定的字段优先级和安全兜底。
- stderr 尾部必须设行数上限，否则单个失败仍可能产生大量输出。
- 不应通过 shell 管道或 `jq` 临时裁剪；那会丢失 Python 内部的阶段语义，并可能掩盖原始退出码。
- 不删除或缩减现有 artifact 内容；本次只改变 compact 模式的终端呈现。
- 所有生产代码修改必须在本设计获书面确认后进行。
