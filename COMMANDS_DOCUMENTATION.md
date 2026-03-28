# 实验命令行工具命令文档

## 命令类概览

| 命令类名称 | 职责 | 主要参数 | 使用实例 |
|-----------|------|----------|----------|
| **InitCommand** | 初始化实验目录结构，生成配置文件，创建必要的子目录和示例文件 | `name`: 实验名称<br>`description`: 实验描述<br>`output_dir`: 输出目录 | `experiment-cli init --name "customer_exp" --description "顾客偏好实验"` |
| **TrainCommand** | 执行模型训练，自动记录参数、指标和模型文件，管理实验生命周期 | `config_path`: 配置文件路径<br>`tracking_uri`: 实验追踪路径 | `experiment-cli train --config "experiment_config.yaml"` |
| **ListCommand** | 列出历史实验，支持按状态过滤和多维度排序，以表格形式展示结果 | `tracking_uri`: 实验追踪路径<br>`status`: 状态过滤器<br>`sort_by`: 排序字段<br>`limit`: 数量限制 | `experiment-cli list --sort-by accuracy --limit 10` |
| **CompareCommand** | 对比多个实验的差异，包括基本信息和性能指标的对比，识别最佳实验 | `experiment_ids`: 要对比的实验ID列表<br>`tracking_uri`: 实验追踪路径 | `experiment-cli compare exp1 exp2 exp3` |
| **CleanCommand** | 清理过期实验数据，保留指定数量的最佳模型，支持模拟删除和强制删除 | `keep_top_n`: 保留的最佳实验数量<br>`metric`: 排序指标<br>`dry_run`: 模拟删除<br>`force`: 强制删除 | `experiment-cli clean --keep-top-n 5 --metric f1_score` |

## 详细命令说明

### 1. InitCommand - 初始化命令

**职责**: 创建完整的实验目录结构，生成默认配置文件，创建示例数据目录和说明文档。

**参数说明**:
- `--name`: 实验名称（默认: "customer_preference_experiment"）
- `--description`: 实验描述（默认: "顾客偏好预测实验"）
- `--output-dir`: 输出目录（默认: 当前目录）

**使用实例**:
```bash
# 基本初始化
experiment-cli init

# 指定实验名称和描述
experiment-cli init --name "mobile_pref_experiment" --description "移动端用户偏好预测"

# 指定输出目录
experiment-cli init --output-dir "./my_experiments"
```

### 2. TrainCommand - 训练命令

**职责**: 加载配置，执行模型训练，自动记录实验参数、指标和模型文件，管理实验状态（创建、运行、完成、失败）。

**参数说明**:
- `--config`: 配置文件路径（默认: "experiment_config.yaml"）
- `--tracking-uri`: 实验数据存储路径（默认: "./experiments"）

**使用实例**:
```bash
# 使用默认配置文件训练
experiment-cli train

# 指定配置文件
experiment-cli train --config "configs/experiment_01.yaml"

# 指定实验追踪路径
experiment-cli train --tracking-uri "/data/experiments"
```

### 3. ListCommand - 列出实验命令

**职责**: 查询并展示历史实验，支持按状态过滤，按时间或性能指标排序，以美观的表格形式展示结果。

**参数说明**:
- `--tracking-uri`: 实验数据存储路径（默认: "./experiments"）
- `--status`: 按状态过滤（可选值: completed, running, failed）
- `--sort-by`: 排序字段（可选值: created_at, updated_at, accuracy, precision, recall, f1_score）
- `--limit`: 限制显示的实验数量

**使用实例**:
```bash
# 列出所有实验
experiment-cli list

# 只显示已完成的实验
experiment-cli list --status completed

# 按准确率排序，显示前10个
experiment-cli list --sort-by accuracy --limit 10

# 按F1分数排序
experiment-cli list --sort-by f1_score
```

### 4. CompareCommand - 对比命令

**职责**: 对比多个实验的性能差异，包括基本信息对比和性能指标对比，自动识别并标记最佳实验。

**参数说明**:
- `experiment_ids`: 要对比的实验ID（至少2个）
- `--tracking-uri`: 实验数据存储路径（默认: "./experiments"）

**使用实例**:
```bash
# 对比两个实验
experiment-cli compare exp_id_1 exp_id_2

# 对比多个实验
experiment-cli compare exp_id_1 exp_id_2 exp_id_3 exp_id_4
```

### 5. CleanCommand - 清理命令

**职责**: 清理过期或性能较差的实验数据，保留指定数量的最佳模型，支持模拟删除以预览结果。

**参数说明**:
- `--keep-top-n`: 保留的最佳实验数量（默认: 10）
- `--metric`: 用于排序的指标（可选值: accuracy, precision, recall, f1_score, auc_roc）
- `--dry-run`: 模拟删除，不实际执行
- `--force`: 强制删除，不提示确认
- `--tracking-uri`: 实验数据存储路径（默认: "./experiments"）

**使用实例**:
```bash
# 保留前10个最佳实验（按准确率）
experiment-cli clean

# 保留前5个最佳实验（按F1分数）
experiment-cli clean --keep-top-n 5 --metric f1_score

# 模拟清理，预览将删除的实验
experiment-cli clean --dry-run

# 强制清理，不提示确认
experiment-cli clean --force
```

## 工作流程示例

```bash
# 1. 初始化实验
experiment-cli init --name "customer_pref_v1" --description "第一版顾客偏好预测实验"

# 2. 修改配置文件（编辑 experiment_config.yaml）

# 3. 开始训练
experiment-cli train

# 4. 查看实验结果
experiment-cli list --sort-by f1_score

# 5. 对比多个实验
experiment-cli compare exp_id_1 exp_id_2

# 6. 定期清理，保留最佳模型
experiment-cli clean --keep-top-n 10 --metric f1_score
```
