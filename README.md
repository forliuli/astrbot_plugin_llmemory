# LL类神经记忆情绪系统

AstrBot 用的记忆插件。不接云端、不用向量数据库、不配任何模型也能转。

和常见向量记忆的区别：它把聊天碎成节点落库，真正取回靠节点之间那些会自己长、也会自己变淡的边，不靠一条条算相似度。

## 安装

1. 把整个文件夹放进 `AstrBot/data/plugins/`，在面板里启用。
2. 装依赖（可选，但不装会少一层检索）：

```
pip install -r requirements.txt
```

numpy / onnxruntime / tokenizers。面板装插件时一般会自动装依赖；没自动装的话，到 `AstrBot/data/plugins/astrbot_plugin_llmemory/` 下面手动跑一遍上面这行。缺了不挡插件加载，只是向量那层关掉，检索退成时间窗。
3. 重启 AstrBot。首次启动会在后台下向量模型（约 90 MB），下完自己接上。

模型地址面板里能改，默认给了 hf-mirror 和 modelscope 两个镜像，一个不通会试下一个。校验值留着的话会校 sha256，填空就跳过校验。下不下来都不影响使用。

## 装上之后不用配

默认全自动。聊天照常聊，节点、边、情绪都在后台走。

面板里能动的几块：

- `enable` 情绪模块，默认开
- `judge_mode` 默认 `rule`，纯规则判断，不需要模型。选 `model` 才用 `model_url` / `model_key` / `model_name`；留空则退到 AstrBot 已配的 provider
- `affect_tone` 让心情影响说话（默认开），`affect_recall` 让心情影响想起来的旧事（默认关）
- `decay_*` 边权随时间变淡：几小时压一次、多少天细一半
- `reflect_*` 事后回看几句（默认关）
- `drift_enable` 神游、`wake_enable` 自唤醒（默认关；开了有静默时段、每小时和每天的上限、两句之间的最小间隔）
- `auto_persona` / `auto_diary` / `auto_profile` 每天从原始记录自己蒸一版人格、日记、画像（默认全关，要模型）
- `data_dir` / `writer_dir` 数据和人设文件放哪；`writer_provider` 写这些用哪个模型

## 指令

`/情绪` 看一眼当前六维状态
`/写人格` `/写日记` `/写画像` 立刻写一份（按当前配置）

## 装之前请留意

- 这个记忆系统没做短期上下文库，**平台的上下文长度别设太小**。压过来的话太少，节点就稀疏，图也长不出东西来。
- 数据落在 `data/plugin_data/astrbot_plugin_llmemory/`，框架给的目录，可以用 `data_dir` 挪走。要备份就备份那里，`nodes.json`、`graph.npz`、`neurostate.json` 是主干。
- 各模块都能单独关，全关时对原来的对话零影响。怕吵就把神游、自唤醒、反思都留着关。
- 用到的是 AstrBot 通用事件和 `unified_msg_origin`，没绑特定平台。

## 目录

```
main.py               插件入口、配置、指令
_conf_schema.json     面板配置项
llmemory/
  nodes.py            节点：内容、向量、时间、来源
  graph.py            图与赫布边权、exp(-age/tau) 变淡
  vec.py              向量那层（可缺）
  recall.py           取回：入口节点 + 沿边走一跳
  writer.py           落库
  muse.py             神游
  wake.py             自唤醒、长梦
  reflect.py          反思
  model_fetch.py      向量模型下载（.part + sha256 + os.replace）
  affect/             六维情绪，规则判官与模型判官
models/tokenizer.json 分词表
```

## 反馈

有问题提到 Issues。部分使用 ai 制作。
