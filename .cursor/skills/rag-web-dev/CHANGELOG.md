# RAG Web System 开发指南 Skill 版本说明

## 0.2（与项目 v0.4 同步）

- **项目结构**：补充 `docs/`、`docs/preview/`、`README.md`、`.gitignore`；模块注释体现仅答知识库、优化建议、指标目标、成本/效率提示。
- **核心配置**：参数表拆为「应用与基础设施」「Embedding 与 LLM」「RAG 核心参数」，覆盖 `config.py` 全部字段。
- **API**：补充 `GET /` 重定向到 `/ui/`。
- **环境搭建**：启动命令区分 macOS/Linux 与 Windows，Windows 注明 WinError 10013 时改用 `127.0.0.1`。
- **扩展场景**：8.4 增加优化建议与前端目标步骤；新增 8.5「新增前端参数时」补全 `info` 浮窗。
- **引用**：增加 `docs/release-notes-v0.4.md`。
- **第 12 节**：项目经验总结（v0.4 迭代）— 成本/效率透明、指标目标可配置、仅答知识库、无上下文不调 LLM、setuptools 包发现、Windows host、功能预览。

## 0.1

- 初始版本：项目概览、环境搭建、项目结构、核心参数、检索策略、API、测试、扩展场景、Docker、环境检测、报错排查、引用文件索引。
